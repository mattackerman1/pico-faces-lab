"""Sweep Block 4/5 attention-input int8 scales and validate trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from capstone_01_adaln_single import (
    PROJECT,
    decode,
    image_metrics,
    load_decoder,
    load_released_model,
    seeded_noise,
    write_rgb_png,
)
from capstone_02_quant_damage import (
    InstrumentedFakeDiT,
    build_audit_states,
    metric,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.inference_mode()
def evaluate(model, data, batch_size):
    outputs = []
    for start in range(0, len(data["states"]), batch_size):
        sl = slice(start, start + batch_size)
        model.time_indices = data["time_indices"][sl]
        model.path_ids = data["path_ids"][sl]
        outputs.append(model(data["states"][sl], data["times"][sl], data["labels"][sl]))
    return torch.cat(outputs)


@torch.inference_mode()
def sample(model, initial, labels, schedule, guidance):
    z = initial.clone()
    null_label = model.m.n_classes if hasattr(model, "m") else model.n_classes
    for time_value, next_time in zip(schedule, list(schedule[1:]) + [0.0]):
        t = torch.full((len(z),), float(time_value))
        conditional = model(z, t, labels)
        if guidance is None:
            velocity = conditional
        else:
            null = torch.full_like(labels, null_label)
            unconditional = model(z, t, null)
            velocity = unconditional + guidance * (conditional - unconditional)
        z = z - float(time_value - next_time) * velocity
    return z


def triple_grid(reference, baseline, tuned, columns=8):
    arrays = [tensor.permute(0, 2, 3, 1).cpu().numpy() for tensor in (reference, baseline, tuned)]
    rows = []
    for start in range(0, len(reference), columns):
        for array in arrays:
            tiles = list(array[start:start + columns])
            if len(tiles) < columns:
                tiles.extend([np.zeros_like(tiles[0])] * (columns - len(tiles)))
            rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="m3_long_cfg")
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--multipliers4", default="1,1.05,1.10,1.20,1.35,1.50")
    parser.add_argument("--multipliers5", default="1,1.05,1.10,1.20,1.35,1.50")
    parser.add_argument("--out", type=Path, default=PROJECT / "device-output" / "capstone-02-scale-sweep")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    teacher, checkpoint, export_cfg, _ = load_released_model(args.model)
    schedule = [float(value) for value in export_cfg["schedule"]]
    calib_path = PROJECT / "upstream" / "pico-faces" / export_cfg["calib"]
    scales = dict(np.load(calib_path).items())
    data = build_audit_states(teacher, schedule, args.trajectories)

    float_outputs = []
    with torch.inference_mode():
        for start in range(0, len(data["states"]), args.batch_size):
            sl = slice(start, start + args.batch_size)
            float_outputs.append(teacher(data["states"][sl], data["times"][sl], data["labels"][sl]))
    float_outputs = torch.cat(float_outputs)

    multipliers4 = tuple(float(value) for value in args.multipliers4.split(","))
    multipliers5 = tuple(float(value) for value in args.multipliers5.split(","))
    rows = []
    best = None
    for multiplier4 in multipliers4:
        for multiplier5 in multipliers5:
            fake = InstrumentedFakeDiT(
                teacher, scales, checkpoint["cfg"].get("act", "relu2"), export_cfg.get("drop_attn", [])
            )
            fake.s["att_in.4"] *= multiplier4
            fake.s["att_in.5"] *= multiplier5
            fake.record_names = {"block.4.attn.input", "block.5.attn.input"}
            fake.record_detail = False
            candidate = evaluate(fake, data, args.batch_size)
            overall = metric(float_outputs, candidate)
            early = data["time_indices"] <= 1
            early_metrics = metric(float_outputs[early], candidate[early])
            local4 = fake.local["block.4.attn.input"].result()
            local5 = fake.local["block.5.attn.input"].result()
            row = {
                "block4_multiplier": multiplier4,
                "block5_multiplier": multiplier5,
                "velocity_mse": overall["mse"],
                "velocity_relative_rmse": overall["relative_rmse"],
                "velocity_cosine": overall["cosine"],
                "early_velocity_mse": early_metrics["mse"],
                "block4_clip_percent": local4["clipped_percent"],
                "block5_clip_percent": local5["clipped_percent"],
                "block4_local_relative_rmse": local4["relative_rmse"],
                "block5_local_relative_rmse": local5["relative_rmse"],
            }
            rows.append(row)
            if best is None or row["velocity_mse"] < best["velocity_mse"]:
                best = row
            print(
                f"B4 {multiplier4:.2f} B5 {multiplier5:.2f}: "
                f"MSE {overall['mse']:.6f}; clips {local4['clipped_percent']:.3f}%/"
                f"{local5['clipped_percent']:.3f}%"
            )
    write_csv(args.out / "scale-sweep.csv", rows)

    baseline_fake = InstrumentedFakeDiT(
        teacher, scales, checkpoint["cfg"].get("act", "relu2"), export_cfg.get("drop_attn", [])
    )
    tuned_fake = InstrumentedFakeDiT(
        teacher, scales, checkpoint["cfg"].get("act", "relu2"), export_cfg.get("drop_attn", [])
    )
    tuned_fake.s["att_in.4"] *= best["block4_multiplier"]
    tuned_fake.s["att_in.5"] *= best["block5_multiplier"]
    baseline_fake.record = False; tuned_fake.record = False

    labels = torch.repeat_interleave(torch.arange(4), 8)
    seeds = [20_000 + 1_000 * label + member for label in range(4) for member in range(8)]
    initial = seeded_noise(seeds, teacher.z_ch, teacher.z_hw)
    decoder, mean, std = load_decoder(args.model)
    trajectory_report = {}
    for mode, guidance in (("plain", None), ("cfg4", 4.0)):
        float_latents = sample(teacher, initial, labels, schedule, guidance)
        baseline_latents = sample(baseline_fake, initial, labels, schedule, guidance)
        tuned_latents = sample(tuned_fake, initial, labels, schedule, guidance)
        float_images = decode(decoder, mean, std, float_latents)
        baseline_images = decode(decoder, mean, std, baseline_latents)
        tuned_images = decode(decoder, mean, std, tuned_latents)
        trajectory_report[mode] = {
            "baseline_latent_relative_error": (
                (baseline_latents - float_latents).flatten(1).norm(dim=1)
                / float_latents.flatten(1).norm(dim=1).clamp_min(1e-12)
            ).mean().item(),
            "tuned_latent_relative_error": (
                (tuned_latents - float_latents).flatten(1).norm(dim=1)
                / float_latents.flatten(1).norm(dim=1).clamp_min(1e-12)
            ).mean().item(),
            "baseline_image_metrics": image_metrics(float_images, baseline_images),
            "tuned_image_metrics": image_metrics(float_images, tuned_images),
        }
        write_rgb_png(
            args.out / f"{mode}-float-baseline-tuned.png",
            triple_grid(float_images, baseline_images, tuned_images),
        )

    report = {
        "model": args.model,
        "multipliers4": multipliers4,
        "multipliers5": multipliers5,
        "best_velocity_candidate": best,
        "trajectory_evaluation": trajectory_report,
        "interpretation_note": "Best candidate selected only by teacher-state velocity MSE; complete trajectories are the decision gate.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("best velocity candidate:", best)
    for mode, result in trajectory_report.items():
        print(
            f"{mode}: latent {result['baseline_latent_relative_error']:.4f} -> "
            f"{result['tuned_latent_relative_error']:.4f}; pixel MAE "
            f"{result['baseline_image_metrics']['mean_absolute_pixel_error']:.2f} -> "
            f"{result['tuned_image_metrics']['mean_absolute_pixel_error']:.2f}"
        )


if __name__ == "__main__":
    main()
