"""First-pass quantization damage map for the released Pico Faces DiT.

Runs identical plain and CFG-4 sampling states through the floating-point
checkpoint and the repository's fake-deployment int8 graph. It records local
activation-grid damage, cumulative residual divergence, static weight error,
and block-level oracle repairs that identify causally important components.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from capstone_01_adaln_single import PROJECT, load_released_model
from train.common.sincos import timestep_embedding
from quant.fake_quant import prep_scales, qdq_w


PATH_NAMES = {0: "plain", 1: "cfg_conditional", 2: "cfg_null"}


@dataclass
class Running:
    count: int = 0
    sq_reference: float = 0.0
    sq_error: float = 0.0
    abs_error: float = 0.0
    dot: float = 0.0
    sq_candidate: float = 0.0
    clipped: int = 0
    grid_values: int = 0
    grid_abs_sum: float = 0.0
    max_grid_abs: float = 0.0

    def add(self, reference, candidate, grid=None, clipped=None):
        r = reference.detach().double()
        q = candidate.detach().double()
        e = q - r
        self.count += r.numel()
        self.sq_reference += r.square().sum().item()
        self.sq_error += e.square().sum().item()
        self.abs_error += e.abs().sum().item()
        self.dot += (r * q).sum().item()
        self.sq_candidate += q.square().sum().item()
        if clipped is not None:
            self.clipped += int(clipped.detach().sum().item())
        if grid is not None:
            g = grid.detach().double().abs()
            self.grid_values += g.numel()
            self.grid_abs_sum += g.sum().item()
            self.max_grid_abs = max(self.max_grid_abs, g.max().item())

    def result(self):
        ref_rms = math.sqrt(self.sq_reference / max(self.count, 1))
        err_rms = math.sqrt(self.sq_error / max(self.count, 1))
        denom = math.sqrt(self.sq_reference * self.sq_candidate)
        return {
            "values": self.count,
            "reference_rms": ref_rms,
            "error_rms": err_rms,
            "relative_rmse": err_rms / max(ref_rms, 1e-30),
            "mean_absolute_error": self.abs_error / max(self.count, 1),
            "cosine": self.dot / max(denom, 1e-30),
            "clipped_percent": 100.0 * self.clipped / max(self.count, 1),
            "mean_grid_utilization_percent": (
                100.0 * self.grid_abs_sum / max(self.grid_values * 127.0, 1.0)
            ),
            "maximum_grid_magnitude": self.max_grid_abs,
        }


class InstrumentedFakeDiT:
    def __init__(self, model, scales, act="relu2", drop_attn=()):
        self.m = model
        self.s = prep_scales(scales, next(model.parameters()).device)
        self.drop_attn = frozenset(drop_attn)
        self.act_sq = {
            b: act == "relu2" and self.s[f"fc2_in.{b}"].dim() == 1
            for b in range(len(model.blocks))
        }
        self.repair_group = None
        self.record = True
        self.record_names = None
        self.record_detail = True
        self.time_indices = None
        self.path_ids = None
        self.local = defaultdict(Running)
        self.local_by_time_path = defaultdict(Running)
        self.snapshots = []

    def repaired(self, name):
        return self.repair_group is not None and name.startswith(self.repair_group)

    def qa(self, name, x, scale, dim=-1):
        if self.repaired(name):
            return x
        view = scale
        if scale.dim() == 1:
            shape = [1] * x.dim()
            shape[dim] = -1
            view = scale.view(shape)
        limit = 127.0 * view
        clipped = x.abs() > limit
        xc = torch.minimum(torch.maximum(x, -limit), limit)
        grid = torch.round(xc / view)
        q = grid * view
        if self.record and (self.record_names is None or name in self.record_names):
            self.local[name].add(x, q, grid, clipped)
            if self.record_detail:
                for ti in torch.unique(self.time_indices).tolist():
                    for pi in torch.unique(self.path_ids).tolist():
                        chosen = (self.time_indices == ti) & (self.path_ids == pi)
                        if chosen.any():
                            self.local_by_time_path[(name, int(ti), int(pi))].add(
                                x[chosen], q[chosen], grid[chosen], clipped[chosen]
                            )
        return q

    def qw(self, name, weight):
        return weight if self.repaired(name) else qdq_w(weight)

    def __call__(self, z, t, y):
        m, s = self.m, self.s
        batch = len(z)
        self.snapshots = []
        xin = self.qa("embed.z", m.patchify(z), s["z"])
        x = F.linear(xin, self.qw("embed.weight", m.embed.weight), m.embed.bias) + m.pos
        c = m.t_mlp(timestep_embedding(t, m.t_dim)) + m.y_emb(y)

        for b, block in enumerate(m.blocks):
            s1, b1, g1, s2, b2, g2 = block.mod(c)[:, None].chunk(6, dim=-1)
            if b not in self.drop_attn:
                prefix = f"block.{b}.attn"
                attn = block.attn
                heads, head_dim = attn.heads, attn.hd
                xa = self.qa(
                    prefix + ".input",
                    block.norm1(x) * (1 + s1) + b1,
                    s[f"att_in.{b}"],
                )
                qkv = F.linear(
                    xa, self.qw(prefix + ".qkv_weight", attn.qkv.weight), attn.qkv.bias
                )
                tokens = xa.shape[1]
                q, k, v = qkv.view(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
                q = self.qa(prefix + ".q_pre", q, s[f"qk_pre.{b}"])
                k = self.qa(prefix + ".k_pre", k, s[f"qk_pre.{b}"])
                v = self.qa(prefix + ".v", v, s[f"v.{b}"])
                q = self.qa(prefix + ".q_post", attn.q_norm(q), s[f"qk_post.{b}"], dim=1)
                k = self.qa(prefix + ".k_post", attn.k_norm(k), s[f"qk_post.{b}"], dim=1)
                probabilities = torch.softmax(
                    q @ k.transpose(-2, -1) / math.sqrt(head_dim), dim=-1
                )
                probabilities = self.qa(
                    prefix + ".softmax",
                    probabilities,
                    torch.tensor(1.0 / 127.0, device=z.device),
                )
                proposal = (probabilities @ v).transpose(1, 2).reshape(batch, tokens, -1)
                proposal = self.qa(prefix + ".output", proposal, s[f"att_out.{b}"])
                x = x + g1 * F.linear(
                    proposal,
                    self.qw(prefix + ".proj_weight", attn.proj.weight),
                    attn.proj.bias,
                )

            prefix = f"block.{b}.mlp"
            xm = self.qa(
                prefix + ".input",
                block.norm2(x) * (1 + s2) + b2,
                s[f"fc1_in.{b}"],
            )
            h = F.linear(
                xm,
                self.qw(prefix + ".fc1_weight", block.mlp[0].weight),
                block.mlp[0].bias,
            )
            if self.act_sq[b]:
                h = block.mlp[1](h)
                h = self.qa(prefix + ".relu2_output", h, s[f"fc2_in.{b}"])
            else:
                h = self.qa(prefix + ".activation_input", h, s[f"act_in.{b}"])
                h = block.mlp[1](h)
                h = self.qa(prefix + ".activation_output", h, s[f"fc2_in.{b}"])
            x = x + g2 * F.linear(
                h,
                self.qw(prefix + ".fc2_weight", block.mlp[2].weight),
                block.mlp[2].bias,
            )
            self.snapshots.append(x.detach())

        sf, bf = m.final_mod(c)[:, None].chunk(2, dim=-1)
        xf = self.qa("final.input", m.final_norm(x) * (1 + sf) + bf, s["final_in"])
        out = F.linear(xf, self.qw("final.weight", m.final.weight), m.final.bias)
        return m.unpatchify(out)


@torch.inference_mode()
def build_audit_states(model, schedule, trajectories):
    if trajectories % 20:
        raise ValueError("trajectories must be divisible by 20")
    generator = torch.Generator(device="cpu").manual_seed(90210)
    states, times, labels, time_indices, path_ids = [], [], [], [], []
    next_times = list(schedule[1:]) + [0.0]

    plain_count = trajectories
    z = torch.randn(plain_count, model.z_ch, model.z_hw, model.z_hw, generator=generator)
    y = torch.arange(plain_count) % (model.n_classes + 1)
    for ti, (tv, tn) in enumerate(zip(schedule, next_times)):
        t = torch.full((plain_count,), float(tv))
        states.append(z.clone()); times.append(t); labels.append(y.clone())
        time_indices.append(torch.full((plain_count,), ti)); path_ids.append(torch.zeros(plain_count, dtype=torch.long))
        z = z - float(tv - tn) * model(z, t, y)

    cfg_count = trajectories
    z = torch.randn(cfg_count, model.z_ch, model.z_hw, model.z_hw, generator=generator)
    y = torch.arange(cfg_count) % model.n_classes
    null = torch.full_like(y, model.n_classes)
    for ti, (tv, tn) in enumerate(zip(schedule, next_times)):
        t = torch.full((cfg_count,), float(tv))
        for which_y, path in ((y, 1), (null, 2)):
            states.append(z.clone()); times.append(t); labels.append(which_y.clone())
            time_indices.append(torch.full((cfg_count,), ti)); path_ids.append(torch.full((cfg_count,), path, dtype=torch.long))
        vc, vn = model(z, t, y), model(z, t, null)
        z = z - float(tv - tn) * (vn + 4.0 * (vc - vn))

    return {
        "states": torch.cat(states), "times": torch.cat(times),
        "labels": torch.cat(labels), "time_indices": torch.cat(time_indices),
        "path_ids": torch.cat(path_ids),
    }


def metric(reference, candidate):
    r, q = reference.double(), candidate.double()
    error = q - r
    return {
        "mse": error.square().mean().item(),
        "relative_rmse": error.square().mean().sqrt().item() / max(r.square().mean().sqrt().item(), 1e-30),
        "cosine": F.cosine_similarity(r.flatten(1), q.flatten(1), dim=1).mean().item(),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="m3_long_cfg")
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=PROJECT / "device-output" / "capstone-02-quant-damage")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    teacher, checkpoint, export_cfg, checkpoint_path = load_released_model(args.model)
    schedule = [float(v) for v in export_cfg["schedule"]]
    scales = dict(np.load(Path(__file__).parent / "upstream" / "pico-faces" / export_cfg["calib"]).items())
    fake = InstrumentedFakeDiT(
        teacher, scales, checkpoint["cfg"].get("act", "relu2"), export_cfg.get("drop_attn", [])
    )
    data = build_audit_states(teacher, schedule, args.trajectories)
    count = len(data["states"])
    print(f"audit calls: {count} ({args.trajectories} plain and {args.trajectories} CFG trajectories)")

    float_outputs, fake_outputs = [], []
    cumulative = defaultdict(Running)
    captured = []
    hooks = [
        block.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
        for block in teacher.blocks
    ]
    with torch.inference_mode():
        for start in range(0, count, args.batch_size):
            sl = slice(start, start + args.batch_size)
            captured.clear()
            reference = teacher(data["states"][sl], data["times"][sl], data["labels"][sl])
            float_snaps = list(captured)
            fake.time_indices = data["time_indices"][sl]
            fake.path_ids = data["path_ids"][sl]
            candidate = fake(data["states"][sl], data["times"][sl], data["labels"][sl])
            float_outputs.append(reference); fake_outputs.append(candidate)
            for b, (fr, fq) in enumerate(zip(float_snaps, fake.snapshots)):
                cumulative[b].add(fr, fq)
    for hook in hooks:
        hook.remove()
    float_outputs, fake_outputs = torch.cat(float_outputs), torch.cat(fake_outputs)
    baseline = metric(float_outputs, fake_outputs)
    baseline_by_time_path = []
    for ti, time_value in enumerate(schedule):
        for pi, path_name in PATH_NAMES.items():
            chosen = (data["time_indices"] == ti) & (data["path_ids"] == pi)
            baseline_by_time_path.append({
                "time_index": ti, "time": time_value, "path": path_name,
                **metric(float_outputs[chosen], fake_outputs[chosen]),
            })
    write_csv(args.out / "baseline-time-path.csv", baseline_by_time_path)

    local_rows = []
    for name, running in fake.local.items():
        local_rows.append({"boundary": name, **running.result()})
    local_rows.sort(key=lambda row: row["relative_rmse"], reverse=True)
    write_csv(args.out / "local-boundaries.csv", local_rows)

    detail_rows = []
    for (name, ti, pi), running in fake.local_by_time_path.items():
        detail_rows.append({
            "boundary": name, "time_index": ti, "time": schedule[ti],
            "path": PATH_NAMES[pi], **running.result(),
        })
    detail_rows.sort(key=lambda row: (row["boundary"], row["time_index"], row["path"]))
    write_csv(args.out / "boundary-time-path.csv", detail_rows)

    weight_rows = []
    weights = {"embed.weight": teacher.embed.weight, "final.weight": teacher.final.weight}
    for b, block in enumerate(teacher.blocks):
        weights.update({
            f"block.{b}.attn.qkv_weight": block.attn.qkv.weight,
            f"block.{b}.attn.proj_weight": block.attn.proj.weight,
            f"block.{b}.mlp.fc1_weight": block.mlp[0].weight,
            f"block.{b}.mlp.fc2_weight": block.mlp[2].weight,
        })
    for name, weight in weights.items():
        running = Running(); running.add(weight, qdq_w(weight))
        weight_rows.append({"weight": name, **running.result()})
    weight_rows.sort(key=lambda row: row["relative_rmse"], reverse=True)
    write_csv(args.out / "weights.csv", weight_rows)

    cumulative_rows = [{"block": b, **cumulative[b].result()} for b in range(len(teacher.blocks))]
    write_csv(args.out / "cumulative-blocks.csv", cumulative_rows)

    groups = ["embed"]
    for b in range(len(teacher.blocks)):
        groups.extend([f"block.{b}.attn", f"block.{b}.mlp"])
    groups.append("final")
    # The first local scan historically points to the middle attention blocks.
    # Repair their individual grids and matrices to separate a conspicuous
    # local discrepancy from a boundary with real downstream influence.
    for b in (4, 5):
        prefix = f"block.{b}.attn"
        groups.extend(prefix + suffix for suffix in (
            ".input", ".qkv_weight", ".q_pre", ".k_pre", ".v",
            ".q_post", ".k_post", ".softmax", ".output", ".proj_weight",
        ))
    repair_rows = []
    fake.record = False
    for group in groups:
        fake.repair_group = group
        repaired = []
        with torch.inference_mode():
            for start in range(0, count, args.batch_size):
                sl = slice(start, start + args.batch_size)
                fake.time_indices = data["time_indices"][sl]
                fake.path_ids = data["path_ids"][sl]
                repaired.append(fake(data["states"][sl], data["times"][sl], data["labels"][sl]))
        repaired_metrics = metric(float_outputs, torch.cat(repaired))
        recovery = 100.0 * (baseline["mse"] - repaired_metrics["mse"]) / max(baseline["mse"], 1e-30)
        repair_rows.append({"component": group, "repaired_mse": repaired_metrics["mse"], "mse_recovery_percent": recovery, "repaired_relative_rmse": repaired_metrics["relative_rmse"], "repaired_cosine": repaired_metrics["cosine"]})
        print(f"repair {group:16s}: {recovery:+7.2f}% final-MSE recovery")
    repair_rows.sort(key=lambda row: row["mse_recovery_percent"], reverse=True)
    write_csv(args.out / "oracle-repairs.csv", repair_rows)

    report = {
        "model": args.model, "checkpoint": str(checkpoint_path),
        "calibration": export_cfg["calib"], "schedule": schedule,
        "plain_trajectories": args.trajectories, "cfg4_trajectories": args.trajectories,
        "evaluated_dit_calls": count, "baseline_fake_quant_vs_float": baseline,
        "baseline_by_time_path": baseline_by_time_path,
        "top_local_relative_error": local_rows[:12],
        "top_local_clipping": sorted(local_rows, key=lambda row: row["clipped_percent"], reverse=True)[:12],
        "cumulative_blocks": cumulative_rows, "top_weight_error": weight_rows[:12],
        "oracle_repairs": repair_rows,
        "limitations": [
            "Fake quantization mirrors QAT boundaries but is not the byte-exact integer simulator.",
            "Oracle repairs bypass all weight and activation grids in one component; interactions can make recovery non-additive or negative.",
            "This first map covers the DiT only, not the VAE decoder.",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"baseline fake-int vs float: MSE {baseline['mse']:.6f}, relative RMSE {baseline['relative_rmse']:.4f}, cosine {baseline['cosine']:.4f}")
    print(f"report: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
