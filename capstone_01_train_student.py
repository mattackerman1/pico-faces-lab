"""Conditioning-only AdaLN-single student pilot.

The released DiT is the frozen teacher. The student copies its complete body,
but replaces the eight block-specific modulation MLPs at runtime with one
shared MLP plus learned per-block offsets. Only those new conditioning
parameters are trained. A cached synthetic pool of teacher sampling states is
used for distillation; the downloaded FFHQ holdout remains evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from capstone_01_adaln_single import (
    PROJECT,
    collect_modulations,
    fit_additive,
    load_released_model,
)
from capstone_01_velocity_loss import CLASS_NAMES
from train.common.sincos import timestep_embedding


class SharedAdaLNStudent(nn.Module):
    def __init__(self, body, schedule):
        super().__init__()
        self.body = body
        self.schedule = tuple(float(value) for value in schedule)
        dim = body.blocks[0].mod[1].in_features
        depth = len(body.blocks)
        self.shared_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.block_offsets = nn.Parameter(torch.zeros(depth, 6, dim))

        # Averaging the block-specific linear layers gives the exact mean
        # modulation function because every block uses the same SiLU input.
        with torch.no_grad():
            weights = torch.stack([block.mod[1].weight for block in body.blocks])
            biases = torch.stack([block.mod[1].bias for block in body.blocks])
            self.shared_mod[1].weight.copy_(weights.mean(dim=0))
            self.shared_mod[1].bias.copy_(biases.mean(dim=0))

            mods = collect_modulations(body, self.schedule)
            _, offsets, _ = fit_additive(mods)
            self.block_offsets.copy_(offsets)

        for parameter in self.body.parameters():
            parameter.requires_grad_(False)
        self.shared_mod.requires_grad_(True)
        self.block_offsets.requires_grad_(True)
        self.body.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The copied body has no dropout needed for this pilot. Keep all its
        # normalization/batch semantics fixed while training conditioning.
        self.body.eval()
        self.shared_mod.train(mode)
        return self

    def conditioning(self, t, labels):
        body = self.body
        c = body.t_mlp(timestep_embedding(t, body.t_dim))
        c = c + body.y_emb(labels)
        return c

    def forward(self, z, t, labels):
        body = self.body
        x = body.embed(body.patchify(z)) + body.pos
        c = self.conditioning(t, labels)
        shared = self.shared_mod(c).reshape(len(z), 6, -1)

        for block_index, block in enumerate(body.blocks):
            values = shared + self.block_offsets[block_index]
            s1, b1, g1, s2, b2, g2 = (
                value[:, None] for value in values.unbind(dim=1)
            )
            x = x + g1 * block.attn(block.norm1(x) * (1 + s1) + b1)
            x = x + g2 * block.mlp(block.norm2(x) * (1 + s2) + b2)

        scale, bias = body.final_mod(c)[:, None].chunk(2, dim=-1)
        x = body.final(body.final_norm(x) * (1 + scale) + bias)
        return body.unpatchify(x)


@torch.inference_mode()
def build_teacher_pool(teacher, schedule, path, trajectories, batch_size):
    if path.exists():
        print(f"loading cached teacher pool: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    if trajectories % (teacher.n_classes + 1):
        raise ValueError("trajectory count must be divisible by five conditions")
    generator = torch.Generator(device="cpu").manual_seed(1701)
    labels = torch.arange(trajectories) % (teacher.n_classes + 1)
    permutation = torch.randperm(trajectories, generator=generator)
    labels = labels[permutation]
    noise = torch.randn(
        trajectories, teacher.z_ch, teacher.z_hw, teacher.z_hw,
        generator=generator,
    )
    next_times = list(schedule[1:]) + [0.0]
    states, targets, times, saved_labels = [], [], [], []

    for start in range(0, trajectories, batch_size):
        z = noise[start:start + batch_size].clone()
        y = labels[start:start + batch_size]
        for time_value, next_time in zip(schedule, next_times):
            t = torch.full((len(z),), float(time_value))
            velocity = teacher(z, t, y)
            states.append(z.clone())
            targets.append(velocity.clone())
            times.append(t)
            saved_labels.append(y.clone())
            z = z - float(time_value - next_time) * velocity
        print(f"teacher trajectories {min(start + batch_size, trajectories)}/{trajectories}")

    pool = {
        "states": torch.cat(states),
        "targets": torch.cat(targets),
        "times": torch.cat(times),
        "labels": torch.cat(saved_labels),
    }
    torch.save(pool, path)
    print(f"saved teacher pool: {path}")
    return pool


def per_sample_metrics(target, prediction):
    target_flat = target.flatten(1)
    error_flat = (prediction - target).flatten(1)
    return {
        "mse": error_flat.square().mean(dim=1),
        "relative_rmse": (
            error_flat.norm(dim=1) / target_flat.norm(dim=1).clamp_min(1e-12)
        ),
        "cosine": F.cosine_similarity(prediction.flatten(1), target_flat, dim=1),
    }


@torch.inference_mode()
def evaluate_teacher_pool(student, pool, indices, batch_size):
    totals = []
    student.eval()
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        prediction = student(
            pool["states"][chosen], pool["times"][chosen], pool["labels"][chosen]
        )
        totals.append((prediction - pool["targets"][chosen]).square().mean().item())
    return float(np.mean(totals))


@torch.inference_mode()
def heldout_rows(teacher, student, latents, labels, schedule, batch_size):
    generator = torch.Generator(device="cpu").manual_seed(424242)
    noise = torch.randn(latents.shape, generator=generator)
    time_indices = torch.arange(len(latents)) % len(schedule)
    rows = []

    for time_index, time_value in enumerate(schedule):
        members = torch.nonzero(time_indices == time_index).flatten()
        for offset in range(0, len(members), batch_size):
            chosen = members[offset:offset + batch_size]
            clean, eps = latents[chosen], noise[chosen]
            t = torch.full((len(chosen),), float(time_value))
            mixed = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * eps
            target = eps - clean
            y = labels[chosen]
            original = teacher(mixed, t, y)
            candidate = student(mixed, t, y)
            original_metrics = per_sample_metrics(target, original)
            student_metrics = per_sample_metrics(target, candidate)
            for batch_index, sample_index in enumerate(chosen.tolist()):
                rows.append({
                    "sample": sample_index,
                    "time_index": time_index,
                    "time": float(time_value),
                    "class": int(y[batch_index]),
                    "class_name": CLASS_NAMES[int(y[batch_index])],
                    "teacher_mse": original_metrics["mse"][batch_index].item(),
                    "student_mse": student_metrics["mse"][batch_index].item(),
                    "teacher_relative_rmse": original_metrics["relative_rmse"][batch_index].item(),
                    "student_relative_rmse": student_metrics["relative_rmse"][batch_index].item(),
                    "teacher_cosine": original_metrics["cosine"][batch_index].item(),
                    "student_cosine": student_metrics["cosine"][batch_index].item(),
                })
    rows.sort(key=lambda row: row["sample"])
    return rows


def summarize_heldout(rows):
    teacher_mse = float(np.mean([row["teacher_mse"] for row in rows]))
    student_mse = float(np.mean([row["student_mse"] for row in rows]))
    return {
        "teacher_mse": teacher_mse,
        "student_mse": student_mse,
        "student_mse_increase_percent": (student_mse / teacher_mse - 1) * 100,
        "teacher_cosine": float(np.mean([row["teacher_cosine"] for row in rows])),
        "student_cosine": float(np.mean([row["student_cosine"] for row in rows])),
    }


def by_time(rows):
    return {
        str(index): summarize_heldout([row for row in rows if row["time_index"] == index])
        for index in sorted({row["time_index"] for row in rows})
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="m3_long_cfg")
    parser.add_argument("--trajectories", type=int, default=80)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--late-weight",
        type=float,
        default=1.0,
        help="relative sampling weight for t <= 0.25 teacher states",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "device-output" / "capstone-01-adaln-single" / "student-pilot",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20261007)
    teacher, checkpoint, export_cfg, checkpoint_path = load_released_model(args.model)
    teacher.requires_grad_(False)
    teacher.eval()
    schedule = [float(value) for value in export_cfg["schedule"]]
    student = SharedAdaLNStudent(teacher, schedule)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"trainable conditioning parameters: {trainable:,}")

    pool_path = args.out / f"teacher-pool-{args.trajectories}.pt"
    pool = build_teacher_pool(
        teacher, schedule, pool_path, args.trajectories, args.batch_size
    )
    count = len(pool["states"])
    split_generator = torch.Generator(device="cpu").manual_seed(99)
    order = torch.randperm(count, generator=split_generator)
    validation_count = count // 5
    pool_valid = order[:validation_count]
    pool_train = order[validation_count:]
    sampling_weights = torch.ones(len(pool_train))
    late_members = pool["times"][pool_train] <= 0.25 + 1e-8
    sampling_weights[late_members] = args.late_weight
    sampling_generator = torch.Generator(device="cpu").manual_seed(20261008)
    print(
        f"training sampler: {late_members.sum().item()}/{len(pool_train)} late states, "
        f"late weight {args.late_weight:g}"
    )

    validation_dir = (
        PROJECT / "device-output" / "capstone-01-adaln-single" / "validation"
    )
    heldout_latents = torch.from_numpy(np.load(validation_dir / "latents.npy")).float()
    source_labels = torch.from_numpy(np.load(validation_dir / "labels.npy")).long()
    heldout_labels = torch.repeat_interleave(source_labels, 2)

    initial_rows = heldout_rows(
        teacher, student, heldout_latents, heldout_labels, schedule, args.batch_size
    )
    initial_summary = summarize_heldout(initial_rows)
    print(
        f"initial heldout: teacher {initial_summary['teacher_mse']:.6f}, "
        f"student {initial_summary['student_mse']:.6f} "
        f"({initial_summary['student_mse_increase_percent']:+.1f}%)"
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )
    log_rows = []
    started = time.time()
    student.train()
    for step in range(1, args.steps + 1):
        sampled_positions = torch.multinomial(
            sampling_weights,
            args.batch_size,
            replacement=True,
            generator=sampling_generator,
        )
        chosen = pool_train[sampled_positions]
        prediction = student(
            pool["states"][chosen], pool["times"][chosen], pool["labels"][chosen]
        )
        loss = F.mse_loss(prediction, pool["targets"][chosen])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            valid_loss = evaluate_teacher_pool(
                student, pool, pool_valid, args.batch_size
            )
            elapsed = time.time() - started
            row = {
                "step": step,
                "train_distill_mse": loss.item(),
                "pool_validation_distill_mse": valid_loss,
                "elapsed_seconds": elapsed,
            }
            log_rows.append(row)
            print(
                f"step {step:4d}: train {loss.item():.6f}, "
                f"pool-valid {valid_loss:.6f}, elapsed {elapsed:.1f}s"
            )
            student.train()

    final_rows = heldout_rows(
        teacher, student, heldout_latents, heldout_labels, schedule, args.batch_size
    )
    final_summary = summarize_heldout(final_rows)
    recovered = (
        (initial_summary["student_mse"] - final_summary["student_mse"])
        / max(initial_summary["student_mse"] - initial_summary["teacher_mse"], 1e-30)
    )
    report = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "trainable_parameters": trainable,
        "teacher_trajectories": args.trajectories,
        "teacher_pool_states": count,
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "late_timestep_sampling_weight": args.late_weight,
        "initial_heldout": initial_summary,
        "final_heldout": final_summary,
        "fraction_of_heldout_mse_gap_recovered": recovered,
        "initial_by_time": by_time(initial_rows),
        "final_by_time": by_time(final_rows),
        "training_log": log_rows,
        "limitation": (
            "Conditioning-only pilot distilled plain teacher velocities from "
            "synthetic teacher trajectories; transformer body stayed frozen."
        ),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    with (args.out / "heldout-final.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    torch.save(
        {
            "shared_mod": student.shared_mod.state_dict(),
            "block_offsets": student.block_offsets.detach(),
            "model": args.model,
            "schedule": schedule,
            "report": report,
        },
        args.out / "student-conditioning.pt",
    )

    print("\nfinal heldout result")
    print(
        f"  teacher MSE {final_summary['teacher_mse']:.6f}\n"
        f"  student MSE {final_summary['student_mse']:.6f}\n"
        f"  student penalty {final_summary['student_mse_increase_percent']:+.1f}%\n"
        f"  original approximation gap recovered {recovered:.1%}\n"
        f"  cosine teacher {final_summary['teacher_cosine']:.4f}, "
        f"student {final_summary['student_cosine']:.4f}"
    )
    print(f"report: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
