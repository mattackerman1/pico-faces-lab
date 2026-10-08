# Lesson 11 — Rebuild the released model artifact

## Goal

Recreate the fast model's `model.bin`, `rf_cfg.h`, and golden outputs from its
released quantization-aware-training checkpoint and frozen calibration, then
compare the rebuilt blob byte-for-byte with the released reference.

This lesson is divided into two bites:

- **11A:** understand the artifact-building pipeline and what “folding” means;
- **11B:** create an isolated Python environment, run the fold/export process,
  and verify the rebuilt files.

## 11A — Training, folding, compiling, and running are different operations

We have now encountered four distinct stages:

```text
training
   learned examples -> adjustable model weights

folding/exporting
   trained checkpoint + frozen calibration + configuration
       -> deployment-ready model.bin + rf_cfg.h + goldens

compiling
   C source + rf_cfg.h -> executable machine instructions

inference
   executable + model.bin + seed/settings -> RGB image
```

Lesson 11 performs the second operation. We are **not retraining** the model.

## Inputs to the fast-model fold

The released inputs are under `upstream/pico-faces`.

### `checkpoints/m3_long_cfg/dit_qat.pt`

This approximately 19 MB PyTorch checkpoint contains the DiT weights after
quantization-aware fine-tuning. The weights are still stored in a convenient
training representation rather than the final packed deployment format.

### `checkpoints/m3_long_cfg/vae_final.pt`

This approximately 39 MB checkpoint contains the VAE, including the decoder
that turns the final latent into RGB pixels. The exporter needs only the
decoder portion for generation.

### `checkpoints/m3_long_cfg/calib_cfg.npz`

This contains the frozen activation ranges collected from representative
sampling trajectories, including the guided path. Those ranges determine the
integer scales used by the deployment model.

### `checkpoints/m3_long_cfg/latent_stats.npz`

This small file contains the latent mean and standard deviation. The exporter
folds latent de-normalization into the decoder's first convolution.

### `models/m3_long_cfg/*.yaml`

The YAML configuration files describe the DiT, VAE, sampling schedule,
quantization choices, CFG weights, and artifact paths.

## What “folding” means

Here, folding means performing calculations offline so the Pico does not have
to repeat them during every generated image.

Important examples include:

- quantizing trained weights to symmetric int8 grids;
- converting biases into integer accumulator units;
- calculating the integer multipliers and shifts used for requantization;
- combining batch-normalization parameters with VAE convolution weights;
- precomputing AdaLN conditioning for the finite time/class combinations;
- incorporating AdaLN gates into branch-output rescaling;
- incorporating each Euler step size into the velocity-output rescaling;
- incorporating latent mean and standard deviation into the decoder; and
- building lookup tables and packing arrays into the exact blob layout.

The general pattern is:

```text
calculation that depends only on fixed trained data
                         |
                         v
                 perform it on the PC once
                         |
                         v
          store its result in deployment-ready form
```

This trades a little more offline preparation for less Pico computation and a
simpler runtime.

## Why the calibration must remain frozen

The QAT student adapted its weights to particular simulated integer grids.
Those grids were derived from the saved calibration. If we recalibrate after
QAT, we change the scales—and therefore the integer codes and the meaning of
those codes—without giving the model another opportunity to adapt.

For a reproducible released build, the checkpoint and calibration are a
matched pair:

```text
QAT checkpoint + the calibration used during QAT
```

Using that pair is also necessary if we expect our exported bytes to match the
released `model.bin` exactly.

## Expected outputs

Running `quant/fold.py --model m3_long_cfg` writes:

```text
artifacts/m3_long_cfg/export/md.pkl
artifacts/m3_long_cfg/export/model.bin
artifacts/m3_long_cfg/export/rf_cfg.h
artifacts/m3_long_cfg/goldens/e2e_trained/golden_*.rgb
```

- `md.pkl` is an intermediate Python representation of the folded model.
- `model.bin` is the packed blob consumed by the C engine.
- `rf_cfg.h` is the compile-time geometry generated for that blob.
- the goldens are produced by the exact Python integer simulator.

After exporting, we will compile the C engine against the newly generated
header, compare C outputs against the newly generated goldens, and finally
compare the rebuilt blob against the released reference blob.

## Environment observation

The `python` command currently visible to this project resolves to Python
3.13.13, even though the originally reported version was 3.14.4. That visible
installation currently has none of the three packages required for folding:
PyTorch, NumPy, and PyYAML.

In 11B we will create a project-local virtual environment rather than modifying
the global Python installation. We only need the folding dependencies; the
full dataset and training dependency set is unnecessary.

## 11A checkpoint

Explain these in your own words:

1. Why is folding not the same as training or inference?
2. What is gained by precomputing fixed conditioning, normalization, and
   requantization calculations on the PC?
3. Why must the released QAT checkpoint be paired with its frozen calibration?
4. What additional assurance do we gain by comparing the rebuilt `model.bin`
   itself, rather than only comparing one generated image?

## 11B — Rebuild result on this Windows host

The local rebuild environment uses Python 3.14.4, NumPy 2.5.3, PyYAML 6.0.3,
and the official CPU-only PyTorch 2.9.0 wheel. Running
`lesson_11b_rebuild.ps1` successfully produced a new 2,567,828-byte
`model.bin`, generated four Python integer-simulator goldens, compiled the C
engine against the new `rf_cfg.h`, and matched every C output to its rebuilt
golden byte-for-byte.

The rebuilt blob did **not** match the released blob:

```text
rebuilt SHA-256:  E50FE675A75C2D0435DADFE3067E60C73EF1F0A2E18F3AC10E218571496C4AD8
released SHA-256: 322A1AF56DCC2C34AD51D7B6952AB5AD3E75ACE8A8CB00AA7A1545243ACBA6B0
```

Both blobs have the same header, geometry, format, and byte length. The first
difference occurs at byte 18,012 inside the precomputed conditioning region;
98,415 of 2,567,828 bytes differ. This is consistent with small differences in
offline floating-point evaluation crossing later integer-rounding boundaries.

The repository documents the release environment as WSL with
`2.9.0-dev+cu129`, but does not pin the exact PyTorch nightly revision. Our
runtime is Windows `2.9.0+cpu`, and the checkpoint itself does not record the
original PyTorch build. Therefore, the release's exact floating-point export
environment is not currently reproducible from the repository alone.

There was also an upstream source inconsistency: cleanup commit `766b4c4`
deleted `rfpaths.py` while the folding scripts still imported it. We restored
the exact helper from the immediately preceding commit in this local checkout.

### What the result proves

- The released checkpoints and calibration can be loaded and folded.
- The exporter produces a structurally valid deployment artifact.
- The Python integer simulator and portable C engine agree exactly for all
  four generated fast-model golden paths.
- The Windows fold is deterministic within its own environment.

### What it does not prove

- It does not reproduce the released artifact byte-for-byte.
- It does not establish that the new blob has exactly the same output or image
  quality as the released blob.
- Reaching the release hash requires the exact original floating-point export
  environment or upstream-provided precomputed conditioning values.
