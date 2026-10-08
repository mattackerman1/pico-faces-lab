"""Feasibility audit for an AdaLN-single-style Pico Faces model.

This does not retrain or modify the released checkpoint. It finds the best
additive approximation

    modulation[block, condition] ~= shared[condition] + block_offset[block]

over every exported timestep/class combination, then measures its effect on
sampling trajectories and decoded images.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent
UPSTREAM = PROJECT / "upstream" / "pico-faces"
sys.path.insert(0, str(UPSTREAM))

import rfpaths  # noqa: E402
from train.common.sincos import timestep_embedding  # noqa: E402
from train.dit.model import build_model  # noqa: E402
from train.vae.model import build_vae  # noqa: E402


COMPONENTS = ("attn_scale", "attn_shift", "attn_gate",
              "mlp_scale", "mlp_shift", "mlp_gate")
CLASS_NAMES = ("female_neutral", "female_smiling",
               "male_neutral", "male_smiling")


def clean_state_dict(state):
    return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def load_released_model(model_name: str):
    export_cfg = rfpaths.cfg(model_name, "export")
    checkpoint_path = Path(rfpaths.resolve(export_cfg["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["cfg"])
    model.load_state_dict(clean_state_dict(checkpoint["ema"]))
    model.eval()
    return model, checkpoint, export_cfg, checkpoint_path


@torch.inference_mode()
def collect_modulations(model, schedule):
    """Return [block, condition, time, six, width] AdaLN outputs."""
    times = torch.tensor(schedule, dtype=torch.float32)
    base = model.t_mlp(timestep_embedding(times, model.t_dim))
    condition_vectors = torch.stack(
        [base + model.y_emb.weight[y] for y in range(model.n_classes + 1)]
    )
    per_block = []
    for block in model.blocks:
        values = torch.stack([block.mod(c) for c in condition_vectors])
        per_block.append(values.reshape(values.shape[0], values.shape[1], 6, -1))
    return torch.stack(per_block)


def fit_additive(modulations):
    """Two-way additive least-squares fit with zero-mean block offsets."""
    shared = modulations.mean(dim=0)
    offsets = (modulations - shared.unsqueeze(0)).mean(dim=(1, 2))
    approximation = shared.unsqueeze(0) + offsets[:, None, None]
    return shared, offsets, approximation


def tensor_metrics(reference, approximation):
    error = approximation - reference
    ref_rms = reference.square().mean().sqrt().item()
    err_rms = error.square().mean().sqrt().item()
    ref_flat = reference.flatten().double()
    app_flat = approximation.flatten().double()
    cosine = torch.nn.functional.cosine_similarity(ref_flat, app_flat, dim=0).item()
    return {
        "reference_rms": ref_rms,
        "error_rms": err_rms,
        "relative_rmse": err_rms / max(ref_rms, 1e-12),
        "mean_absolute_error": error.abs().mean().item(),
        "maximum_absolute_error": error.abs().max().item(),
        "cosine_similarity": cosine,
    }


@torch.inference_mode()
def forward_with_table(model, z, t, labels, table, time_index):
    """DiT forward using a supplied [block,cond,time,6,width] table."""
    x = model.embed(model.patchify(z)) + model.pos
    for block_index, block in enumerate(model.blocks):
        values = table[block_index, labels, time_index]
        s1, b1, g1, s2, b2, g2 = (
            value[:, None] for value in values.unbind(dim=1)
        )
        x = x + g1 * block.attn(block.norm1(x) * (1 + s1) + b1)
        x = x + g2 * block.mlp(block.norm2(x) * (1 + s2) + b2)

    c = model.t_mlp(timestep_embedding(t, model.t_dim)) + model.y_emb(labels)
    scale, bias = model.final_mod(c)[:, None].chunk(2, dim=-1)
    x = model.final(model.final_norm(x) * (1 + scale) + bias)
    return model.unpatchify(x)


@torch.inference_mode()
def velocity(model, z, t_value, labels, table, time_index, guidance):
    batch = z.shape[0]
    t = torch.full((batch,), float(t_value), dtype=z.dtype)

    def run(which_labels):
        if table is None:
            return model(z, t, which_labels)
        return forward_with_table(model, z, t, which_labels, table, time_index)

    conditional = run(labels)
    if guidance is None:
        return conditional
    null_labels = torch.full_like(labels, model.n_classes)
    unconditional = run(null_labels)
    return unconditional + guidance * (conditional - unconditional)


def seeded_noise(seeds, channels, size):
    samples = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        samples.append(torch.randn(channels, size, size, generator=generator))
    return torch.stack(samples)


@torch.inference_mode()
def compare_trajectories(model, table, schedule, guidance):
    labels = torch.arange(model.n_classes, dtype=torch.long)
    seeds = [101, 202, 303, 404]
    initial = seeded_noise(seeds, model.z_ch, model.z_hw)
    reference = initial.clone()
    approximate = initial.clone()
    step_metrics = []
    next_times = list(schedule[1:]) + [0.0]

    for index, (time_value, next_time) in enumerate(zip(schedule, next_times)):
        v_reference = velocity(
            model, reference, time_value, labels, None, index, guidance
        )
        v_approximate = velocity(
            model, approximate, time_value, labels, table, index, guidance
        )
        error = v_approximate - v_reference
        relative = (
            error.flatten(1).norm(dim=1)
            / v_reference.flatten(1).norm(dim=1).clamp_min(1e-12)
        )
        cosine = torch.nn.functional.cosine_similarity(
            v_reference.flatten(1), v_approximate.flatten(1), dim=1
        )
        step_metrics.append({
            "time": float(time_value),
            "mean_relative_velocity_error": relative.mean().item(),
            "mean_velocity_cosine": cosine.mean().item(),
        })
        dt = float(time_value - next_time)
        reference = reference - dt * v_reference
        approximate = approximate - dt * v_approximate

    latent_error = approximate - reference
    latent_relative = (
        latent_error.flatten(1).norm(dim=1)
        / reference.flatten(1).norm(dim=1).clamp_min(1e-12)
    )
    return {
        "seeds": seeds,
        "step_metrics": step_metrics,
        "mean_final_latent_relative_error": latent_relative.mean().item(),
        "per_class_final_latent_relative_error": {
            CLASS_NAMES[index]: value.item()
            for index, value in enumerate(latent_relative)
        },
    }, reference, approximate


def load_decoder(model_name):
    vae_cfg = rfpaths.cfg(model_name, "vae")
    export_cfg = rfpaths.cfg(model_name, "export")
    path = Path(rfpaths.resolve(export_cfg["vae_checkpoint"]))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vae = build_vae(vae_cfg)
    vae.load_state_dict(clean_state_dict(checkpoint["model"]))
    vae.eval()
    stats_path = UPSTREAM / "checkpoints" / model_name / "latent_stats.npz"
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"]).float()[None, :, None, None]
    std = torch.from_numpy(stats["std"]).float()[None, :, None, None]
    return vae.decoder, mean, std


@torch.inference_mode()
def decode(decoder, mean, std, latents):
    images = decoder(latents * std + mean).clamp(-1, 1)
    return ((images + 1) * 127.5).round().to(torch.uint8)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an HWC uint8 RGB image using only the Python standard library."""
    height, width, channels = image.shape
    if image.dtype != np.uint8 or channels != 3:
        raise ValueError("PNG writer requires HWC uint8 RGB data")
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def comparison_grid(reference, approximate):
    """Rows are reference, approximation, and amplified absolute difference."""
    ref = reference.permute(0, 2, 3, 1).cpu().numpy()
    app = approximate.permute(0, 2, 3, 1).cpu().numpy()
    diff = np.minimum(np.abs(ref.astype(np.int16) - app.astype(np.int16)) * 4, 255).astype(np.uint8)
    return np.concatenate(
        [np.concatenate(list(ref), axis=1),
         np.concatenate(list(app), axis=1),
         np.concatenate(list(diff), axis=1)],
        axis=0,
    )


def image_metrics(reference, approximate):
    error = approximate.float() - reference.float()
    mse = error.square().mean().item()
    return {
        "mean_absolute_pixel_error": error.abs().mean().item(),
        "root_mean_square_pixel_error": math.sqrt(mse),
        "maximum_absolute_pixel_error": error.abs().max().item(),
        "psnr_db": float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="m3_long_cfg")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "device-output" / "capstone-01-adaln-single",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model, checkpoint, export_cfg, checkpoint_path = load_released_model(args.model)
    schedule = [float(value) for value in export_cfg["schedule"]]
    modulations = collect_modulations(model, schedule)
    shared, offsets, approximation = fit_additive(modulations)

    report = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "shape": list(modulations.shape),
        "ordinary_modulation_values": int(modulations.numel()),
        "adaln_single_values": int(shared.numel() + offsets.numel()),
        "value_reduction_fraction": 1.0 - (
            shared.numel() + offsets.numel()
        ) / modulations.numel(),
        "overall_modulation_fit": tensor_metrics(modulations, approximation),
        "component_modulation_fit": {
            name: tensor_metrics(modulations[..., index, :], approximation[..., index, :])
            for index, name in enumerate(COMPONENTS)
        },
        "trajectories": {},
    }

    decoder, mean, std = load_decoder(args.model)
    for name, guidance in (("plain", None), ("cfg4", 4.0)):
        trajectory, ref_latents, app_latents = compare_trajectories(
            model, approximation, schedule, guidance
        )
        ref_images = decode(decoder, mean, std, ref_latents)
        app_images = decode(decoder, mean, std, app_latents)
        trajectory["image_metrics"] = image_metrics(ref_images, app_images)
        report["trajectories"][name] = trajectory
        write_rgb_png(
            args.out / f"{name}-reference-approx-diff4x.png",
            comparison_grid(ref_images, app_images),
        )

    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"model: {args.model}")
    print(f"AdaLN tensor shape: {tuple(modulations.shape)}")
    print(
        f"values: {report['ordinary_modulation_values']:,} -> "
        f"{report['adaln_single_values']:,} "
        f"({report['value_reduction_fraction']:.1%} reduction)"
    )
    print("\nmodulation fit")
    for name, metrics in report["component_modulation_fit"].items():
        print(
            f"  {name:12s} relative RMSE {metrics['relative_rmse']:.4f}  "
            f"cosine {metrics['cosine_similarity']:.5f}"
        )
    print("\ntrajectory and image effect")
    for name, result in report["trajectories"].items():
        pixels = result["image_metrics"]
        print(
            f"  {name:5s} final latent relative error "
            f"{result['mean_final_latent_relative_error']:.4f}; "
            f"pixel MAE {pixels['mean_absolute_pixel_error']:.2f}; "
            f"PSNR {pixels['psnr_db']:.2f} dB"
        )
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
