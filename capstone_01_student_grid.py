"""Broad teacher/student image grid for a trained shared-AdaLN student."""

from __future__ import annotations

import argparse
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
from capstone_01_student_images import sample
from capstone_01_train_student import SharedAdaLNStudent


def tiled_pair_grid(teacher, student, columns=8):
    """For each tile row, put teacher images above their student matches."""
    teacher = teacher.permute(0, 2, 3, 1).cpu().numpy()
    student = student.permute(0, 2, 3, 1).cpu().numpy()
    rows = []
    for start in range(0, len(teacher), columns):
        t = list(teacher[start:start + columns])
        s = list(student[start:start + columns])
        if len(t) < columns:
            blank = np.zeros_like(t[0])
            t.extend([blank] * (columns - len(t)))
            s.extend([blank] * (columns - len(s)))
        rows.extend([np.concatenate(t, axis=1), np.concatenate(s, axis=1)])
    return np.concatenate(rows, axis=0)


def per_class_metrics(reference, candidate, labels):
    result = {}
    for label in range(4):
        chosen = labels == label
        result[str(label)] = image_metrics(reference[chosen], candidate[chosen])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds-per-class", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.out / "student-conditioning.pt", map_location="cpu", weights_only=False
    )
    teacher, _, export_cfg, _ = load_released_model(checkpoint["model"])
    schedule = [float(value) for value in export_cfg["schedule"]]
    student = SharedAdaLNStudent(teacher, schedule)
    student.shared_mod.load_state_dict(checkpoint["shared_mod"])
    student.block_offsets.data.copy_(checkpoint["block_offsets"])
    student.eval()

    # Distinct reproducible seeds for each class; class-major order makes each
    # contact sheet easy to inspect from neutral female through smiling male.
    seeds = []
    labels = []
    for label in range(4):
        for member in range(args.seeds_per_class):
            seeds.append(10_000 + 1_000 * label + member)
            labels.append(label)
    labels = torch.tensor(labels, dtype=torch.long)
    initial = seeded_noise(seeds, teacher.z_ch, teacher.z_hw)
    decoder, mean, std = load_decoder(checkpoint["model"])

    modes = (("plain", None), ("cfg2", 2.0), ("cfg4", 4.0), ("cfg6", 6.0))
    report = {
        "seeds_per_class": args.seeds_per_class,
        "seeds": seeds,
        "modes": {},
    }
    for name, guidance in modes:
        teacher_latents = sample(teacher, initial, labels, schedule, guidance)
        student_latents = sample(student, initial, labels, schedule, guidance)
        teacher_images = decode(decoder, mean, std, teacher_latents)
        student_images = decode(decoder, mean, std, student_latents)
        relative = (
            (student_latents - teacher_latents).flatten(1).norm(dim=1)
            / teacher_latents.flatten(1).norm(dim=1).clamp_min(1e-12)
        )
        metrics = image_metrics(teacher_images, student_images)
        report["modes"][name] = {
            "guidance": guidance,
            "mean_final_latent_relative_error": relative.mean().item(),
            "image_metrics": metrics,
            "per_class_image_metrics": per_class_metrics(
                teacher_images, student_images, labels
            ),
        }
        write_rgb_png(
            args.out / f"grid-{name}-teacher-student.png",
            tiled_pair_grid(teacher_images, student_images),
        )
        print(
            f"{name}: latent {relative.mean().item():.4f}; "
            f"pixel MAE {metrics['mean_absolute_pixel_error']:.2f}; "
            f"PSNR {metrics['psnr_db']:.2f} dB"
        )

    (args.out / "grid-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
