"""Fold the selected attention scales and verify exact-integer behavior."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from capstone_01_adaln_single import (
    PROJECT,
    decode,
    image_metrics,
    load_decoder,
    load_released_model,
    write_rgb_png,
)
from capstone_02_scale_sweep import triple_grid
from quant.export import write_goldens, write_model_bin, write_rf_cfg
from quant.fold import fold
from quant.int_ops import gaussian_clt12, pcg32_init, sat16
from quant.int_sim import IntSim


OUT = PROJECT / "device-output" / "capstone-02-exact-fold"
UPSTREAM = PROJECT / "upstream" / "pico-faces"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_latents(sim, seeds):
    samples = []
    for seed in seeds:
        state = pcg32_init(seed)
        _, noise = gaussian_clt12(state, sim.tokens * sim.pd)
        tokens = sat16(noise.reshape(sim.tokens, sim.pd))
        hwc = sim.unpatchify(tokens).astype(np.float32) / 4096.0
        samples.append(torch.from_numpy(hwc).permute(2, 0, 1))
    return torch.stack(samples)


@torch.inference_mode()
def float_sample(model, initial, labels, schedule, guidance, k_steps=4):
    stride = len(schedule) // k_steps
    indices = list(range(0, len(schedule), stride))
    z = initial.clone()
    for position, index in enumerate(indices):
        time_value = float(schedule[index])
        next_time = float(schedule[indices[position + 1]]) if position + 1 < len(indices) else 0.0
        t = torch.full((len(z),), time_value)
        conditional = model(z, t, labels)
        if guidance is None:
            velocity = conditional
        else:
            null = torch.full_like(labels, model.n_classes)
            unconditional = model(z, t, null)
            velocity = unconditional + guidance * (conditional - unconditional)
        z = z - (time_value - next_time) * velocity
    return z


def sim_images(sim, seeds, labels, guided):
    images = []
    for seed, label in zip(seeds, labels.tolist()):
        image, _ = sim.generate(seed, k_steps=4, cond=label, w_idx=0 if guided else -1)
        images.append(torch.from_numpy(image).permute(2, 0, 1))
    return torch.stack(images)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tuned_dir = OUT / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)

    teacher, _, export_cfg, checkpoint_path = load_released_model("m3_long_cfg")
    source_calib = UPSTREAM / export_cfg["calib"]
    original = dict(np.load(source_calib).items())
    tuned = {key: np.array(value, copy=True) for key, value in original.items()}
    multipliers = {"att_in.4": 1.80, "att_in.5": 1.40}
    changed = {}
    absent_optional_entries = []
    for key, multiplier in multipliers.items():
        for actual in (key, key + "__pc"):
            if actual not in tuned:
                absent_optional_entries.append(actual)
                continue
            before = tuned[actual].copy()
            tuned[actual] = tuned[actual] * multiplier
            changed[actual] = {
                "multiplier": multiplier,
                "before_min": float(np.min(before)), "before_max": float(np.max(before)),
                "after_min": float(np.min(tuned[actual])), "after_max": float(np.max(tuned[actual])),
            }
    tuned_calib = OUT / "calib-attn45-rescaled.npz"
    np.savez(tuned_calib, **tuned)

    md_tuned = fold("m3_long_cfg", str(checkpoint_path), str(tuned_calib), str(tuned_dir))
    tuned_blob = tuned_dir / "model.bin"
    tuned_header = tuned_dir / "rf_cfg.h"
    write_model_bin(md_tuned, str(tuned_blob))
    write_rf_cfg(md_tuned, str(tuned_header))
    golden_dir = tuned_dir / "goldens"
    write_goldens(md_tuned, [1, 2, 3, 4], str(golden_dir), k_steps=4)

    baseline_dir = UPSTREAM / "artifacts" / "m3_long_cfg" / "export"
    baseline_blob = baseline_dir / "model.bin"
    with (baseline_dir / "md.pkl").open("rb") as stream:
        md_baseline = pickle.load(stream)
    baseline_sim, tuned_sim = IntSim(md_baseline), IntSim(md_tuned)

    seeds = [101, 202, 303, 404]
    labels = torch.arange(4, dtype=torch.long)
    initial = initial_latents(baseline_sim, seeds)
    decoder, mean, std = load_decoder("m3_long_cfg")
    schedule = [float(value) for value in export_cfg["schedule"]]
    quality = {}
    for mode, guidance in (("plain", None), ("cfg4", 4.0)):
        float_latents = float_sample(teacher, initial, labels, schedule, guidance)
        float_images = decode(decoder, mean, std, float_latents)
        baseline_images = sim_images(baseline_sim, seeds, labels, guidance is not None)
        tuned_images = sim_images(tuned_sim, seeds, labels, guidance is not None)
        quality[mode] = {
            "baseline_vs_float": image_metrics(float_images, baseline_images),
            "tuned_vs_float": image_metrics(float_images, tuned_images),
            "tuned_vs_baseline": image_metrics(baseline_images, tuned_images),
        }
        write_rgb_png(
            OUT / f"exact-{mode}-float-baseline-tuned.png",
            triple_grid(float_images, baseline_images, tuned_images, columns=4),
        )

    baseline_bytes, tuned_bytes = baseline_blob.read_bytes(), tuned_blob.read_bytes()
    report = {
        "checkpoint": str(checkpoint_path),
        "source_calibration": str(source_calib),
        "derived_calibration": str(tuned_calib),
        "changed_entries": changed,
        "absent_optional_per_channel_entries": absent_optional_entries,
        "baseline_blob": {
            "path": str(baseline_blob), "bytes": len(baseline_bytes), "sha256": sha256(baseline_blob),
        },
        "tuned_blob": {
            "path": str(tuned_blob), "bytes": len(tuned_bytes), "sha256": sha256(tuned_blob),
        },
        "blob_differing_bytes": sum(a != b for a, b in zip(baseline_bytes, tuned_bytes)),
        "header_byte_identical": tuned_header.read_bytes() == (baseline_dir / "rf_cfg.h").read_bytes(),
        "exact_integer_quality": quality,
        "python_goldens": str(golden_dir),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
