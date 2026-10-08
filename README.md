# Pico Faces Lab

This project now has a board-tested result: a custom Raspberry Pi Pico 2
firmware build that generates 128 x 128 RGB faces entirely on the RP2350 and
returns them over USB.

The capstone improved two attention-input quantization scales in the released
`m3_long_cfg` model:

- Block 4 attention input: `1.80x`
- Block 5 attention input: `1.40x`

The tuned model was folded into a new blob, verified against the portable C
engine, built into a UF2, flashed to the physical board, and checked
byte-for-byte against the exact integer simulator on both the plain and CFG
paths.

## Shipped release

- [Release notes and flashing instructions](releases/pico-faces-attn45-v1/README.md)
- [Custom UF2](releases/pico-faces-attn45-v1/pico_faces_m3_long_cfg_attn45.uf2)
- [Packaged ZIP](releases/pico-faces-attn45-v1.zip)
- [Full capstone record](CAPSTONE.md)
- [Project checkpoint](CHECKPOINT-2026-10-07.md)

Custom UF2 SHA-256:

```text
22716999BEA9771210007334C0501F8B8BD51F277F95ED4A0525022CD398084D
```

The custom firmware is currently running on the Pico 2.

## Verified result

| Path | Board time | Board CRC32 | Exact simulator match |
| --- | ---: | --- | --- |
| Plain, seed 42, class 1, 4 steps | 3,401 ms | `b693c574` | Yes, all 49,152 bytes |
| CFG 4, seed 42, class 1, 4 steps | 5,344 ms | `94ee4f2c` | Yes, all 49,152 bytes |

Across the four-seed exact-integer evaluation, the tuning improved mean pixel
error versus the floating-point model from `3.08` to `2.33` on plain sampling
and from `6.75` to `5.97` with CFG 4.

## Run the USB GUI

Clone the repository and its pinned upstream dependency, then install the two
GUI dependencies:

```powershell
git clone --recurse-submodules https://github.com/mattackerman1/pico-faces-lab.git
cd pico-faces-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gui.txt
```

Connect the Pico 2 and run:

```powershell
.\launch_gui.ps1
```

The GUI simulates an LCD and buttons while the connected Pico performs the
actual generation.

## What is here

- `lessons/` — the completed step-by-step learning path
- `PROGRESS.md` — lesson checkpoints and experimental notes
- `CAPSTONE.md` — AdaLN-single and quantization investigations
- `device-output/` — experimental data, exact-fold artifacts, builds, and
  physical-board verification
- `tools/` and `capstone_*.py` — host tools and experiments
- `upstream/pico-faces/` — the upstream implementation under its MIT license
- `releases/` — the final distributable firmware package

## Architecture

```text
random noise (16 x 16 x 8 latent)
              |
              v
  flow-matching DiT, 1-8 Euler steps
  + optional classifier-free guidance
              |
              v
       small VAE decoder
              |
              v
      128 x 128 RGB image
              |
              v
           USB serial
```

The deployment uses int8 weights, lookup tables for the small fixed set of
time/class conditions, weights stored in flash and read through SRAM, and both
Cortex-M33 cores.

## Origin and license

This learning project is based on Tim's
[AI Image Generation on an RP2350 Microcontroller](https://cpldcpu.github.io/2026/08/28/ai-image-generation-on-a-rp-pico-2-microcontroller/)
and the accompanying
[`cpldcpu/pico-faces`](https://github.com/cpldcpu/pico-faces) repository.
Upstream code and derived firmware remain subject to the included MIT license.
