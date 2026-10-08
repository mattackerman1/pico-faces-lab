"""Generate teacher/student comparison images for the conditioning pilot."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from capstone_01_adaln_single import (
    PROJECT,
    comparison_grid,
    decode,
    image_metrics,
    load_decoder,
    load_released_model,
    seeded_noise,
    write_rgb_png,
)
from capstone_01_train_student import SharedAdaLNStudent


@torch.inference_mode()
def predicted_velocity(model, z, time_value, labels, guidance):
    t = torch.full((len(z),), float(time_value))
    conditional = model(z, t, labels)
    if guidance is None:
        return conditional
    null = torch.full_like(labels, model.body.n_classes if hasattr(model, "body") else model.n_classes)
    unconditional = model(z, t, null)
    return unconditional + guidance * (conditional - unconditional)


@torch.inference_mode()
def sample(model, initial, labels, schedule, guidance):
    z = initial.clone()
    next_times = list(schedule[1:]) + [0.0]
    for time_value, next_time in zip(schedule, next_times):
        velocity = predicted_velocity(model, z, time_value, labels, guidance)
        z = z - float(time_value - next_time) * velocity
    return z


def main():
    out = PROJECT / "device-output" / "capstone-01-adaln-single" / "student-pilot"
    checkpoint_path = out / "student-conditioning.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher, _, export_cfg, _ = load_released_model(checkpoint["model"])
    schedule = [float(value) for value in export_cfg["schedule"]]
    student = SharedAdaLNStudent(teacher, schedule)
    student.shared_mod.load_state_dict(checkpoint["shared_mod"])
    student.block_offsets.data.copy_(checkpoint["block_offsets"])
    student.eval()

    labels = torch.arange(4, dtype=torch.long)
    initial = seeded_noise([101, 202, 303, 404], teacher.z_ch, teacher.z_hw)
    decoder, mean, std = load_decoder(checkpoint["model"])
    report = {}

    for name, guidance in (("plain", None), ("cfg4", 4.0)):
        teacher_latents = sample(teacher, initial, labels, schedule, guidance)
        student_latents = sample(student, initial, labels, schedule, guidance)
        teacher_images = decode(decoder, mean, std, teacher_latents)
        student_images = decode(decoder, mean, std, student_latents)
        latent_relative = (
            (student_latents - teacher_latents).flatten(1).norm(dim=1)
            / teacher_latents.flatten(1).norm(dim=1).clamp_min(1e-12)
        )
        report[name] = {
            "mean_final_latent_relative_error": latent_relative.mean().item(),
            "image_metrics": image_metrics(teacher_images, student_images),
        }
        write_rgb_png(
            out / f"trained-{name}-teacher-student-diff4x.png",
            comparison_grid(teacher_images, student_images),
        )

    (out / "image-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for name, result in report.items():
        print(
            f"{name}: latent relative error "
            f"{result['mean_final_latent_relative_error']:.4f}; "
            f"pixel MAE {result['image_metrics']['mean_absolute_pixel_error']:.2f}; "
            f"PSNR {result['image_metrics']['psnr_db']:.2f} dB"
        )


if __name__ == "__main__":
    main()
