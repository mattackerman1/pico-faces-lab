# Pico Faces attention-scale extension v1

This firmware is an experimental extension of Tim B.'s **Pico Faces** project:

- [Original article: AI Image Generation on an RP2350 Microcontroller](https://cpldcpu.github.io/2026/08/28/ai-image-generation-on-a-rp-pico-2-microcontroller/)
- [Original repository: `cpldcpu/pico-faces`](https://github.com/cpldcpu/pico-faces)

Tim designed and trained the model and created its quantization pipeline, C
inference engine, and RP2350 firmware. This build keeps those components and
changes only two attention-input calibration scales in the released
`m3_long_cfg` configuration:

| Activation | Original scale | New scale | Multiplier |
| --- | ---: | ---: | ---: |
| DiT Block 4 attention input | 7.5036 | 13.5064 | 1.80x |
| DiT Block 5 attention input | 6.2364 | 8.7309 | 1.40x |

The wider ranges remove early-timestep clipping that was identified by a
quantization-damage audit. Nothing about the trained weights, architecture,
VAE, sampler, classes, CFG calculation, or inference engine was changed.

## Measured image impact

Against the same floating-point model on four fixed seeds/classes at four
Euler steps:

| Mode | Metric | Original fold | Rescaled fold | Change |
| --- | --- | ---: | ---: | ---: |
| Plain | Mean pixel error | 3.0804 | **2.3269** | 24.5% lower |
| Plain | PSNR | 35.73 dB | **37.95 dB** | +2.22 dB |
| CFG 4 | Mean pixel error | 6.7495 | **5.9700** | 11.5% lower |
| CFG 4 | PSNR | 28.89 dB | **29.65 dB** | +0.76 dB |

The change preserves the same generated subjects and class behavior while
reducing the integer model's deviation from Tim's floating-point generator.
It is a subtle fidelity improvement, not a new model or a claim that every
image is subjectively better.

## Contents

- `pico_faces_m3_long_cfg_attn45.uf2` — custom board firmware
- `rollback/pico_faces_m3_long_cfg_original.uf2` — untouched upstream firmware
- `firmware-report.json` — machine-readable build and verification record
- `board.png` — image returned by the physical board in the CFG test
- `SHA256SUMS.txt` — integrity hashes
- `LICENSE` — Tim's upstream MIT license

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

The `-v` flag verifies the write and `-x` reboots into the application.

## Physical-board verification

Both tests used seed 42, class 1, and four Euler steps:

| Path | Time | Device/host CRC32 | Exact simulator match |
| --- | ---: | --- | --- |
| Plain | 3,401 ms | `b693c574` | all 49,152 bytes |
| CFG 4 | 5,344 ms | `94ee4f2c` | all 49,152 bytes |

The embedded tuned blob occurs exactly once in the raw firmware binary. The
portable C engine also matched Python exact-integer output byte-for-byte for
four additional cases across plain and CFG 4/6/8 paths.

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

This is an independent extension, not an official `cpldcpu/pico-faces`
release. The full methodology and limitations are documented in the parent
repository's `README.md` and `CAPSTONE.md`.
