# Pico Faces attention-scale firmware v1

This is a custom Raspberry Pi Pico 2 build of the upstream `m3_long_cfg`
face generator. It changes only two calibrated attention-input quantization
scales before folding the model:

- Block 4: `1.80x`
- Block 5: `1.40x`

The purpose is to prevent early-step clipping in noisy latents without making
the remaining int8 grid unnecessarily coarse.

## Contents

- `pico_faces_m3_long_cfg_attn45.uf2` — custom board firmware
- `rollback/pico_faces_m3_long_cfg_original.uf2` — original upstream firmware
- `firmware-report.json` — machine-readable build and verification record
- `board.png` — image returned by the physical board in the CFG test
- `SHA256SUMS.txt` — integrity hashes
- `LICENSE` — upstream MIT license

## Flash the custom firmware

Manual BOOTSEL method:

1. Disconnect the Pico 2.
2. Hold its `BOOTSEL` button while reconnecting USB.
3. Release the button when the `RP2350` drive appears.
4. Copy `pico_faces_m3_long_cfg_attn45.uf2` to that drive.
5. Wait for the drive to disappear and the serial port to return.

With the installed Pico SDK tools, from this directory:

```powershell
. "$HOME\.pico-sdk\picorc.ps1"
picotool load -f -v -x .\pico_faces_m3_long_cfg_attn45.uf2
```

The `-v` flag asks picotool to verify the write and `-x` reboots into the
application.

## Physical-board verification

Both tests used seed 42, class 1, and four Euler steps:

| Path | Time | Device/host CRC32 | Exact simulator match |
| --- | ---: | --- | --- |
| Plain | 3,401 ms | `b693c574` | All 49,152 bytes |
| CFG 4 | 5,344 ms | `94ee4f2c` | All 49,152 bytes |

The embedded tuned blob occurs exactly once in the raw firmware binary. The
portable C engine also matched Python exact-integer output byte-for-byte for
four additional seeds across plain and CFG 4/6/8 paths.

## Quality result

On four matched seeds/classes, versus the floating-point reference:

| Mode | Baseline pixel MAE | Tuned pixel MAE | Baseline PSNR | Tuned PSNR |
| --- | ---: | ---: | ---: | ---: |
| Plain | 3.0804 | 2.3269 | 35.73 dB | 37.95 dB |
| CFG 4 | 6.7495 | 5.9700 | 28.89 dB | 29.65 dB |

This is evidence of improvement, not a broad quality benchmark: the exact
evaluation set is deliberately small.

## Roll back

Flash `rollback/pico_faces_m3_long_cfg_original.uf2` using the same procedure.
Its expected SHA-256 is listed in `SHA256SUMS.txt`.

## Build identity

- Board: Raspberry Pi Pico 2
- Platform: RP2350 ARM-S
- Pico SDK: 2.2.0
- Compiler: Arm GNU Toolchain 14.2.1
- Custom UF2 size: 5,275,648 bytes
- Embedded model SHA-256:
  `9e99dc2d0f105d37c9b7426b24b30597a23a384cb32a41dc81526028422227b9`

Full experimental history and reproducibility notes are in the parent
project's `CAPSTONE.md` and `CHECKPOINT-2026-10-07.md`.
