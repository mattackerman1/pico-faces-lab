# Progress log

Update this file at the end of each lesson. Short, imperfect notes are useful.

## Current status

- Current lesson: Complete — custom quantization-improved firmware shipped
- Project track: Desktop-first, then board-first
- Board: Raspberry Pi Pico 2 (RP2350), connected
- USB data cable: Connected and confirmed data-capable
- Python version: 3.14.4
- Optional VGA hardware: none planned; use USB output
- Optional CUDA GPU: unknown
- Primary goal: embedded AI—working physical demo plus ML and embedded-systems understanding

## Lesson checkpoints

- [x] Lesson 00 — scope and equipment decided
- [x] Lesson 01 — firmware flashed and serial port found
- [x] Lesson 02 — first PNG generated
- [x] Lesson 03 — controls compared
- [x] Lesson 04 — latent-space budget understood
- [x] Lesson 05 — flow-matching demo completed
- [x] Lesson 06 — sampling trajectory inspected
- [x] Lesson 07 — DiT tensor shapes traced
- [x] Lesson 08 — conditioning and CFG explored
- [x] Lesson 09 — int8 error measured
- [x] Lesson 10 — desktop C output verified
- [x] Lesson 11 — artifact rebuilt
- [x] Lesson 12 — firmware architecture mapped
- [x] Lesson 13 — board benchmark completed
- [x] Lesson 14 — capstone selected
- [x] Lesson 15 — capstone validated
- [x] Lesson 16 — project packaged

## Observations and questions

Add dated notes here as we work.

## Topics to revisit

- How to choose among model architectures for a constrained embedded target.
- How to choose activation functions, including quality, training behavior,
  quantization friendliness, sparsity, and inference cost.

### 2026-10-03

- Begin on the desktop while the board is ordered.
- Initial hardware target: official Raspberry Pi Pico 2 H.
- First output path will be USB-CDC to the host-side PNG viewer, not VGA.
- Longer-term direction: make the result more self-contained with local input
  and/or a compact display after the reference build is working.
- Lesson 04 revealed that VAE, DiT, latent channels, and patch/token vocabulary
  need to be introduced before memory calculations. The lesson was rewritten
  with those foundations.
- Lesson 04 checkpoint completed: random noise enters the DiT, becomes a
  finished latent over repeated refinement steps, and the VAE decoder converts
  it to RGB pixels. Follow-up clarified that latent channels are learned values
  rather than named colors, and that each 2 x 2 x 8 patch contains 32 values.
- Lesson 05 checkpoint arithmetic is correct. Remaining focus: Euler sampling
  repeatedly queries the trained velocity-predicting function and applies a
  small update; it does not replay one velocity stored during training.
- Lesson 06 checkpoint completed: the predicted velocity matches the latent's
  shape so the sampler can scale and add it elementwise; `1 + 0.25*(-3) = 0.25`;
  only the final latent is sent to the VAE decoder.
- Lesson 07 shape trace understood. Follow-up focus: 128 is a chosen model
  width; ReLU-squared produces sparse activations; and the DiT outputs velocity
  while the sampler—not the DiT—constructs the next latent.
- Lesson 07 Euler check completed: `2 + 0.25*(-4) = 1`. Follow-up clarified
  that weights are fixed learned parameters, while activations are temporary
  input-dependent values passed between layers. Sparse zeros remain values,
  although optimized arithmetic may skip work that cannot contribute.
- Lesson 08 was split into 08A (classes, time, RMSNorm, AdaLN-Zero) and 08B
  (classifier-free guidance) after the combined version introduced too much
  jargon at once.
- Lesson 08A follow-up: attention and MLP are the two residual branches; a gate
  scales a branch's proposed change. AdaLN scale/shift/gate operate inside each
  DiT call and are distinct from CFG's null output, conditional output, and
  guidance weight. CFG wraps two AdaLN-conditioned DiT calls.
- Lesson 08A completed. Final distinction: CFG evaluates the same DiT twice;
  `w` is a CFG multiplier rather than an AdaLN gate. RMSNorm normalizes, while
  AdaLN modulates normalized activations and gates residual branch changes.
- Lesson 08B completed. CFG replaces a separate classifier's steering signal
  with the difference between the same DiT's conditional and unconditional
  velocity predictions. Higher guidance generally strengthens class adherence
  while reducing diversity and potentially exaggerating artifacts.
- Lesson 09 completed. Quantization maps real values onto an integer grid using
  a scale, trading rounding resolution against clipping range. Calibration must
  cover representative sampling states and both CFG paths. Per-channel scales
  prevent large-range channels from erasing small-range channels. Wider
  accumulators prevent overflow during multiply-accumulate; they are not a
  second, wider quantization scale. QAT starts from a trained float model and
  uses a frozen float teacher to preserve the learned function while adapting
  the student to deployed integer grids.
- Lesson 10A completed. The C engine is the reusable procedure, `model.bin`
  contains the learned quantized content, `rf_cfg.h` supplies compile-time
  geometry, and raw golden RGB bytes provide a deterministic regression test.
- Lesson 10B setup: installed Zig 0.17 after the larger WinLibs installer
  stalled. The portable C engine compiled successfully for `m3_long_cfg`.
  Seed 1 at four steps generated 49,152 bytes and matched the released golden
  RGB byte-for-byte (CRC32 `b32e6e63`, SHA-256
  `F2B9AAC0661A9E5452D22829A82F7C6CF5CE6B971F4F1607ACFF48F1CE106222`).
- Lesson 10 completed. The compiler turns the C inference procedure into
  machine instructions while learned parameters remain in `model.bin`.
  `rf_cfg.h` fixes array capacities and loop geometry so SRAM needs are known;
  the loader rejects a blob whose recorded geometry does not match. One
  byte-exact golden verifies a substantial end-to-end path but cannot prove
  every seed, class, step count, guidance setting, or model variant.
- Lesson 11A paused before its checkpoint when the Pico 2 arrived. Resume from
  `lessons/11-rebuild-model-blob.md` after the initial board bring-up.
- Hardware track resumed at Lesson 01: identify the exact board, confirm the
  USB cable enumerates it in BOOTSEL mode, then flash the released fast-model
  UF2 and identify its Windows COM port.
- Lesson 01 completed. The board is an official Raspberry Pi Pico 2. BOOTSEL
  appeared as drive `E:` with label `RP2350` and device name `RP2350BOOT`; its
  `INFO_UF2.TXT` identified `UF2 Bootloader v1.0`, model `Raspberry Pi RP2350`,
  and board ID `RP2350`. Flashed `pico_faces_m3_long_cfg.uf2`; the boot drive
  disappeared on reboot and the firmware enumerated as `COM3`.
- Lesson 02 completed. Created project-local `.venv-viewer` using Python 3.14.4
  with PySerial 3.5 and Pillow 12.3.0. Requested seed 42, class 1, four Euler
  steps, and CFG weight 4 over `COM3`. The Pico returned a 128 x 128 RGB frame
  in 5,438 ms with CRC32 `82e03d24`; saved as
  `device-output/seed_42.png` and visually verified.

### 2026-10-06

- Lesson 03 Experiment A held seed 42, class 1, and CFG 4 constant while
  changing only the Euler-step count:

  | Steps | Device time | CRC32 |
  | ---: | ---: | --- |
  | 1 | 2,587 ms | `fbbd9970` |
  | 2 | 3,531 ms | `31c70032` |
  | 4 | 5,429 ms | `82e03d24` |
  | 8 | 9,210 ms | `ddc2cf3d` |

- All four host-computed CRCs matched the device CRCs. Images are stored under
  `device-output/lesson03-steps/`. Class-1 smile and rough face structure were
  already visible after one step; face and expression became more coherent by
  four steps while hair/background remained less resolved. Eight steps did not
  look substantially better than four for this seed, demonstrating diminishing
  visual returns despite the increase from 5.429 to 9.210 seconds.
- Lesson 03 Experiment B held seed 42, four steps, and requested CFG 4 constant
  while changing only the condition:

  | Class | Label | Device time | CRC32 |
  | ---: | --- | ---: | --- |
  | 0 | female, neutral | 5,349 ms | `ce487cbc` |
  | 1 | female, smiling | 5,429 ms | `82e03d24` |
  | 2 | male, neutral | 5,359 ms | `6408f4c3` |
  | 3 | male, smiling | 5,344 ms | `e9cf73c0` |
  | 4 | unconditional/null | 3,463 ms | `7b5a9684` |

- Classes 0–3 used two DiT evaluations per step for CFG. Class 4 is the null
  condition, so CFG would compare the null prediction with itself; the firmware
  skips guidance and uses one pass, explaining its shorter time. Images are in
  `device-output/lesson03-classes/`. The smiling conditions were more coherent
  than the neutral class-0 output for this seed. Facial structure remained
  similar across conditions, consistent with holding the initial noise fixed.
  The null output appeared more average and less strongly directed.
- Lesson 03 Experiment C held seed 42, four steps, and class 1 constant:

  | Guidance | Device time | CRC32 |
  | --- | ---: | --- |
  | plain | 3,484 ms | `bde6bba1` |
  | CFG 4 | 5,427 ms | `82e03d24` |
  | CFG 6 | 5,438 ms | `50182c71` |
  | CFG 8 | 5,398 ms | `65c426a6` |

- Plain sampling used one DiT pass per step. Every supported CFG weight used
  two passes, so changing guidance strength changed the blend but not the
  computation count. Images are in `device-output/lesson03-cfg/`. Increasing
  guidance strengthened the smile, with little added benefit above CFG 4;
  CFG 6 appeared somewhat less natural, though differences among 4/6/8 were
  modest. Preferred baseline: four steps and CFG 4, about 5.4 seconds on this
  board. Higher guidance remains an equal-cost aesthetic option.
- Lesson 03 completed. One and two steps established recognizable structure,
  four steps substantially improved coherence, and eight showed diminishing
  returns. Conditions visibly steered expression/demographic statistics while
  the fixed seed preserved related structure. CFG improved class adherence at
  the expected cost of a second DiT evaluation per step.
- Lesson 11A completed. Training learns parameters; folding happens afterward
  and converts those learned parameters plus frozen calibration into a
  deployment-ready integer artifact. Inference executes that artifact. Folding
  precomputes only fixed transformations, reducing repeated Pico work. A
  byte-identical blob proves all packed deployment data match the release, but
  engine behavior and configuration paths still require execution tests.
- Lesson 11B partial result: created `.venv-fold` with Python 3.14.4, NumPy
  2.5.3, PyYAML 6.0.3, and CPU PyTorch 2.9.0. Upstream cleanup commit `766b4c4`
  had deleted the still-imported `rfpaths.py`; restored its exact previous
  version locally. Folding produced a structurally valid 2,567,828-byte blob.
  Rebuilt Python simulator and rebuilt C engine matched byte-for-byte for seeds
  1–4. The rebuilt blob did not match the released blob: rebuilt SHA-256
  `E50FE675A75C2D0435DADFE3067E60C73EF1F0A2E18F3AC10E218571496C4AD8`,
  release `322A1AF56DCC2C34AD51D7B6952AB5AD3E75ACE8A8CB00AA7A1545243ACBA6B0`.
  Headers, size, and geometry match; first difference is byte 18,012 in the
  conditioning region and 98,415 bytes differ. Likely cause is the unpinned
  release environment (`2.9.0-dev+cu129` under WSL) versus Windows
  `2.9.0+cpu`; exact nightly revision is absent from checkpoint and repo.
- Lesson 11 closed with a qualified result rather than marked complete: a new
  artifact was rebuilt and internally verified, but it did not reproduce the
  release hash and no custom UF2 was built. Exact release reproduction remains
  an environment-provenance follow-up.
- Lesson 12 started. Source inspection corrected an earlier shorthand: weight
  staging always copies required matrices from XIP flash into reusable SRAM,
  but the checked-in build defaults to synchronous `memcpy`; paced asynchronous
  DMA is an optional `RF_STAGE_DMA=1` build path.
- Lesson 12 completed. The full model remains embedded in XIP flash; the engine
  stages only the current matrices into shared SRAM. Both cores read the same
  staged data and write disjoint token/row ranges, then synchronize—normally no
  merge computation is required because their halves already form one output.
  Phase-exclusive buffers let DiT staging, VAE ping-pong activations, decoder
  scratch, and optional display data reuse the same SRAM at different times.
- Lesson 13 completed. Ran three repetitions of plain and CFG 4 sampling at
  1, 2, 4, and 8 steps with seed 42 and class 1. Every repeated configuration
  had an identical CRC, and observed run-to-run timing ranges were at most
  7 ms. Linear fits were `plain = 1666.5 + 459.1 * steps` ms and
  `CFG4 = 1640.0 + 946.8 * steps` ms. The roughly 1.65-second intercept
  estimates fixed generation work; each plain step adds about 0.46 seconds,
  while each guided step adds about 0.95 seconds because CFG evaluates the DiT
  twice. Four-step CFG 4 remains the baseline at a mean 5,436.7 ms.
- Lesson 14 completed. Selected a desktop GUI that simulates the proposed LCD
  and button interface. The actual Pico 2 will continue to perform PRNG, DiT
  sampling, and VAE decoding; the PC will provide only virtual controls,
  serial transport, and a virtual 128 x 128 display. Success requires visible
  busy/error state, timing and CRC reporting, deterministic repeat generation,
  and a design whose actions can later map cleanly to physical controls.
- Lesson 15 implementation complete; final user-visible check pending. Added a
  responsive Tkinter virtual device with a 128 x 128 LCD, class buttons, seed,
  step and guidance controls, save support, serial status, timing, and CRC
  reporting. It compiles and its protocol self-test passes. A real request via
  its serial client returned the seed-42 baseline from COM3 in 5,429 ms with
  matching CRC `82e03d24`. The GUI launched successfully and remains open for
  the visual Generate-button check.
- Lesson 15 visual check passed: the user confirmed the GUI works well.
- Extended capstone Part 1 recorded in `CAPSTONE.md`: evaluate an
  AdaLN-single-inspired shared time-and-class conditioning function plus
  per-block embeddings. Begin with an additive approximation audit of the
  released float checkpoint before attempting fine-tuning, exporter changes,
  or firmware changes.
- Extended capstone Part 1 initial audit completed. The optimal additive
  shared-condition plus block-offset approximation reduced raw AdaLN values by
  85%, but incurred 22.5% modulation relative RMSE. Across four classes, final
  latent relative error was 45.5% for plain sampling and 55.6% for CFG 4;
  decoded pixel MAE was 18.52 and 25.68 levels respectively. Images retained
  rough identity/class structure but lost substantial contrast and detail.
  Conclusion: the released checkpoint cannot simply have its tables replaced;
  an AdaLN-single student would require distillation/fine-tuning or retraining.
- Extended capstone Part 1 held-out velocity audit completed on 128 FFHQ
  images selected from the original holdout range (256 latents including
  flips). Against the known flow target, assigned-class MSE rose from 1.1246
  to 1.3331 (+18.5%) and null-condition MSE from 1.1277 to 1.3314 (+18.1%).
  Assigned-class target cosine fell from 0.6545 to 0.5698. Damage was
  concentrated near the clean end of sampling: +29.0% MSE at t=0.25 and
  +50.8% at t=0.125. This shows the post-hoc shared-AdaLN approximation is not
  merely different; it is worse on an independent held-out training objective.
- Project checkpoint saved as `CHECKPOINT-2026-10-07.md`. Current recommended
  next step is a conditioning-only shared-AdaLN student pilot using cached
  teacher trajectory states, while preserving the existing held-out subset for
  evaluation. Alternatives are whole-model GPU fine-tuning, the Part 2
  quantization-damage audit, full retraining, or packaging and pausing.
- Extended capstone Part 1 conditioning-only pilot completed. A shared
  time-and-class conditioning MLP plus learned block offsets was trained while
  the complete transformer body remained frozen. Training used 640 cached
  states from 80 balanced teacher trajectories for 600 CPU optimization steps.
  On the untouched held-out set, the shared model's velocity-MSE penalty fell
  from 18.5% before training to 3.2%, recovering 82.8% of the gap to the
  teacher; target cosine improved from 0.5698 to 0.6408 versus 0.6545 for the
  teacher. Late-step penalties fell from 29.0%/50.8% to 5.4%/6.6%.
- Four fixed-seed image comparisons support the numeric result. Plain sampling
  improved from pixel MAE 18.52 and PSNR 20.75 dB to MAE 7.12 and PSNR 27.98
  dB. CFG 4 improved from MAE 25.68 and PSNR 17.46 dB to MAE 11.91 and PSNR
  23.50 dB. Plain outputs are visually close; CFG still amplifies residual
  differences, although tested outputs remain coherent and class-directed.
- Updated decision: shared AdaLN conditioning is now a credible candidate.
  Expand conditioning-only distillation and evaluation, especially late-step
  accuracy and CFG robustness, before considering whole-body fine-tuning or
  changing the exporter, integer format, C engine, or firmware.
- Extended capstone Part 1 expanded conditioning experiment completed. The
  teacher pool grew from 80 to 320 trajectories, training from 600 to 1,600
  steps, and the final two timesteps received 2.5x sampling weight. Held-out
  velocity MSE improved from 1.1605 to 1.1550, reducing the teacher penalty
  from 3.2% to 2.7%; target cosine improved from 0.6408 to 0.6428.
- Late weighting reduced the final two timestep penalties from 5.37%/6.58% to
  2.51%/3.26%, but shifted error into several middle timesteps. On a matched
  32-seed, four-class evaluation, the expanded checkpoint did not improve
  decoded teacher agreement: plain MAE changed 7.11 to 7.58 and CFG-4 MAE
  12.63 to 12.86. All inspected outputs remained coherent and class-directed.
- Lesson from the expanded run: isolated velocity loss and final image
  agreement are related but not interchangeable. Errors at every step feed
  subsequent states, so optimizing late steps too heavily can improve the
  held-out average while slightly worsening complete trajectories. The next
  training comparison should use milder/balanced timestep weighting and
  full-trajectory checkpoint selection, or escalate to limited whole-body
  fine-tuning if the conditioning-only tradeoff persists.
- Extended capstone Part 2 first DiT quantization-damage map completed. It used
  20 plain and 20 CFG-4 trajectories, including conditional and null passes,
  for 480 float/fake-int comparisons. Aggregate velocity relative RMSE was
  8.65%, with cosine 0.9967.
- Oracle repairs ranked Block 4 attention (26.26% final-MSE recovery) and Block
  5 attention (20.81%) as the leading causal sources. Their AdaLN-normalized
  attention-input grids accounted for 19.18% and 14.08%; their softmax grids
  accounted for another 5.00% and 4.19%. Weight quantization was comparatively
  small. Block 7 MLP was the leading MLP source at 10.37%.
- The worst local error was Block 5 pre-normalization Q at 39.7% relative RMSE,
  but repairing it recovered only 1.51% of final error because Q normalization
  removes most of the discrepancy. Local numerical error therefore cannot be
  treated as causal importance.
- Block 4/5 attention input clipping was concentrated at the first two noisy
  timesteps, around 0.8%, and was similar across plain, CFG conditional, and
  CFG null paths. Recommended next quantization experiment: sweep only those
  two input scales before attempting new QAT, mixed precision, or firmware
  changes.
- Extended capstone Part 2 attention-scale sweep completed. After a 36-point
  coarse search and 25-point refinement, the best fake-deployment multipliers
  were 1.80x for Block 4 and 1.40x for Block 5 attention inputs. Aggregate
  clipping fell from 0.284%/0.275% to 0%/0.0016%. Velocity MSE fell 26.4%,
  from 0.005049 to 0.003718, while cosine improved from 0.9967 to 0.9972.
- The scale improvement survived 32 full trajectories. Plain final-latent
  relative error fell 13.40% to 9.65%, pixel MAE 5.78 to 3.89, and PSNR rose
  29.85 to 32.95 dB. CFG-4 latent error fell 22.83% to 18.93%, pixel MAE 9.39
  to 7.32, and PSNR rose 25.61 to 27.74 dB. Visual inspection found coherent,
  class-directed outputs closer to float.
- The curve flattened beyond 1.80x/1.40x, confirming the expected tradeoff:
  once clipping is removed, larger scales only coarsen rounding resolution.
  The change now needs folding and byte-exact integer-simulator validation;
  it is not yet a deployable Pico model or firmware modification.
- Extended capstone Part 2 exact fold completed. The released calibration has
  scalar attention-input entries and no optional per-channel entries, so the
  derived copy changed only `att_in.4` by 1.80x and `att_in.5` by 1.40x. The
  source calibration remained untouched. The tuned blob is 2,567,828 bytes,
  has the same header/geometry, and differs from the same-environment baseline
  in 41,621 bytes.
- Exact integer simulation on four matched seeds/classes confirmed the quality
  direction. Plain pixel MAE versus float fell 3.08 to 2.33 and PSNR rose
  35.73 to 37.95 dB. CFG-4 MAE fell 6.75 to 5.97 and PSNR rose 28.89 to 29.65
  dB. Visual comparisons passed.
- The portable C engine generated seeds 1–4 from the tuned blob and matched
  every Python integer-simulator RGB output byte-for-byte. CRC32 values were
  `785eaf0a`, `3d15e0bf`, `9532f998`, and `d5c19408`, covering CFG 4/6/8 and
  null/plain golden paths. The tuned blob is verified but has not yet been
  built into or flashed as Pico firmware.
- Installed Raspberry Pi's user-local Pico SDK 2.2.0 build bundle, including
  ARM GCC 14.2.1, CMake, Ninja, pioasm, and picotool, plus the matching
  `pico-extras` tag. Built a custom RP2350 ARM-S UF2 with the tuned model blob
  embedded exactly once. Custom UF2 SHA-256 is
  `22716999BEA9771210007334C0501F8B8BD51F277F95ED4A0525022CD398084D`.
- Picotool reset the Pico through USB, wrote and verified the full custom UF2,
  rebooted it, and the board returned on `COM3`. Physical test seed 42, class
  1, four steps, CFG 4 completed in 5,344 ms. Device and host CRC32 both equal
  `94ee4f2c`, and all 49,152 RGB bytes match the tuned exact-integer simulator.
  The custom quantization-improved firmware is now running on the Pico 2.
- Complementary seed-42 plain generation completed in 3,401 ms with matching
  device/host CRC32 `b693c574`; its frame also matched the tuned simulator
  byte-for-byte. This verifies both single-pass and CFG dual-pass board paths.
- Lesson 16 completed. The board-tested custom UF2, original rollback UF2,
  checksums, machine-readable firmware report, sample board output, license,
  and flashing instructions were packaged as `pico-faces-attn45-v1`. A
  portable ZIP was produced, and the project landing page now points to the
  shipped result. The Pico 2 remains on the custom tuned firmware.

