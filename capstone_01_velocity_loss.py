"""Compare original and shared-AdaLN held-out flow-matching velocity loss."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from capstone_01_adaln_single import (
    PROJECT,
    UPSTREAM,
    clean_state_dict,
    collect_modulations,
    fit_additive,
    forward_with_table,
    load_released_model,
)

import rfpaths
from train.vae.model import build_vae


CLASS_NAMES = (
    "female_neutral",
    "female_smiling",
    "male_neutral",
    "male_smiling",
    "null",
)


def load_encoder(model_name: str):
    vae_cfg = rfpaths.cfg(model_name, "vae")
    export_cfg = rfpaths.cfg(model_name, "export")
    checkpoint = torch.load(
        rfpaths.resolve(export_cfg["vae_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    vae = build_vae(vae_cfg)
    vae.load_state_dict(clean_state_dict(checkpoint["model"]))
    vae.eval()
    stats = np.load(
        UPSTREAM / "checkpoints" / model_name / "latent_stats.npz"
    )
    mean = torch.from_numpy(stats["mean"]).float()[None, :, None, None]
    std = torch.from_numpy(stats["std"]).float()[None, :, None, None]
    return vae.encoder, mean, std


@torch.inference_mode()
def encode_validation_images(images_path: Path, cache_path: Path, model_name: str):
    if cache_path.exists():
        print(f"loading cached latents: {cache_path}")
        return torch.from_numpy(np.load(cache_path)).float()

    images = torch.from_numpy(np.load(images_path)).permute(0, 3, 1, 2)
    images = images.float() / 127.5 - 1.0
    # Match upstream validation layout: [image, horizontal flip] per source.
    images = torch.stack((images, images.flip(-1)), dim=1).flatten(0, 1)
    encoder, mean, std = load_encoder(model_name)
    batches = []
    for start in range(0, len(images), 8):
        mu, _ = encoder(images[start:start + 8])
        batches.append(((mu - mean) / std).cpu())
        print(f"encoded {min(start + 8, len(images))}/{len(images)}")
    latents = torch.cat(batches)
    np.save(cache_path, latents.numpy().astype(np.float32))
    return latents


def metric_record(target, prediction):
    error = prediction - target
    target_flat = target.flatten(1)
    pred_flat = prediction.flatten(1)
    error_flat = error.flatten(1)
    return {
        "mse": error_flat.square().mean(dim=1),
        "mae": error_flat.abs().mean(dim=1),
        "relative_rmse": (
            error_flat.norm(dim=1) / target_flat.norm(dim=1).clamp_min(1e-12)
        ),
        "cosine": torch.nn.functional.cosine_similarity(
            prediction.flatten(1), target_flat, dim=1
        ),
    }


def summarize_rows(rows):
    keys = ("original_mse", "approximate_mse", "original_mae", "approximate_mae",
            "original_relative_rmse", "approximate_relative_rmse",
            "original_cosine", "approximate_cosine")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["mse_ratio_approx_over_original"] = (
        summary["approximate_mse"] / max(summary["original_mse"], 1e-30)
    )
    summary["mse_increase_percent"] = (
        (summary["mse_ratio_approx_over_original"] - 1.0) * 100.0
    )
    return summary


@torch.inference_mode()
def evaluate(model, approximation, latents, labels, schedule, label_mode, batch_size):
    generator = torch.Generator(device="cpu").manual_seed(424242)
    noise = torch.randn(latents.shape, generator=generator)
    time_indices = torch.arange(len(latents)) % len(schedule)
    rows = []

    for time_index, time_value in enumerate(schedule):
        members = torch.nonzero(time_indices == time_index).flatten()
        for offset in range(0, len(members), batch_size):
            selected = members[offset:offset + batch_size]
            clean = latents[selected]
            eps = noise[selected]
            chosen_labels = labels[selected]
            if label_mode == "null":
                chosen_labels = torch.full_like(chosen_labels, model.n_classes)
            t = torch.full((len(selected),), float(time_value))
            mixed = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * eps
            target = eps - clean
            original = model(mixed, t, chosen_labels)
            approximate = forward_with_table(
                model, mixed, t, chosen_labels, approximation, time_index
            )
            original_metrics = metric_record(target, original)
            approximate_metrics = metric_record(target, approximate)

            for batch_index, source_index in enumerate(selected.tolist()):
                rows.append({
                    "sample": source_index,
                    "source_image": source_index // 2,
                    "flipped": bool(source_index % 2),
                    "time_index": time_index,
                    "time": float(time_value),
                    "class": int(chosen_labels[batch_index]),
                    "class_name": CLASS_NAMES[int(chosen_labels[batch_index])],
                    "original_mse": original_metrics["mse"][batch_index].item(),
                    "approximate_mse": approximate_metrics["mse"][batch_index].item(),
                    "original_mae": original_metrics["mae"][batch_index].item(),
                    "approximate_mae": approximate_metrics["mae"][batch_index].item(),
                    "original_relative_rmse": original_metrics["relative_rmse"][batch_index].item(),
                    "approximate_relative_rmse": approximate_metrics["relative_rmse"][batch_index].item(),
                    "original_cosine": original_metrics["cosine"][batch_index].item(),
                    "approximate_cosine": approximate_metrics["cosine"][batch_index].item(),
                })
        print(f"{label_mode}: timestep {time_index + 1}/{len(schedule)}")
    rows.sort(key=lambda row: row["sample"])
    return rows


def grouped_summary(rows, key):
    values = sorted({row[key] for row in rows})
    return {
        str(value): summarize_rows([row for row in rows if row[key] == value])
        for value in values
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="m3_long_cfg")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=PROJECT / "device-output" / "capstone-01-adaln-single" / "validation",
    )
    args = parser.parse_args()

    model, checkpoint, export_cfg, checkpoint_path = load_released_model(args.model)
    schedule = [float(value) for value in export_cfg["schedule"]]
    modulations = collect_modulations(model, schedule)
    _, _, approximation = fit_additive(modulations)

    latents = encode_validation_images(
        args.validation_dir / "images.npy",
        args.validation_dir / "latents.npy",
        args.model,
    )
    source_labels = torch.from_numpy(np.load(args.validation_dir / "labels.npy")).long()
    labels = torch.repeat_interleave(source_labels, 2)
    if len(labels) != len(latents):
        raise RuntimeError(f"label/latent mismatch: {len(labels)} vs {len(latents)}")

    all_rows = {}
    report = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "source_images": len(source_labels),
        "validation_latents_including_flips": len(latents),
        "schedule": schedule,
        "data_caveat": (
            "Held-out indices match the original last-1000-image range, but "
            "the subset uses dataset-server JPEG renditions rather than the "
            "byte-identical PNG archive used in training."
        ),
        "evaluations": {},
    }
    for mode in ("assigned", "null"):
        rows = evaluate(
            model, approximation, latents, labels, schedule, mode, args.batch_size
        )
        all_rows[mode] = rows
        report["evaluations"][mode] = {
            "overall": summarize_rows(rows),
            "by_time": grouped_summary(rows, "time_index"),
            "by_class": grouped_summary(rows, "class"),
        }

    report_path = args.validation_dir / "velocity-loss-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for mode, rows in all_rows.items():
        csv_path = args.validation_dir / f"velocity-loss-{mode}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nheld-out velocity loss")
    for mode, result in report["evaluations"].items():
        overall = result["overall"]
        print(
            f"  {mode:8s} original MSE {overall['original_mse']:.6f}; "
            f"approx {overall['approximate_mse']:.6f}; "
            f"increase {overall['mse_increase_percent']:.1f}%"
        )
        print(
            f"           original cosine {overall['original_cosine']:.4f}; "
            f"approx {overall['approximate_cosine']:.4f}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
