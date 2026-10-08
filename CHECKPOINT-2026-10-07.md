# Project checkpoint — 2026-10-07

## Working system

- Official Raspberry Pi Pico 2 on `COM3`, running the released
  `pico_faces_m3_long_cfg` firmware.
- Generation is deterministic and host/device CRC checks pass.
- Preferred baseline: class 1, seed 42, four Euler steps, CFG 4.
- Baseline device time: about 5.44 seconds.
- Measured timing model:
  - plain: `1666.5 + 459.1 × steps` ms
  - CFG 4: `1640.0 + 946.8 × steps` ms
- The Tkinter capstone GUI is complete and visually validated. It provides a
  virtual LCD, class buttons, seed/step/guidance controls, busy/error state,
  timing, CRC verification, and image saving. The Pico still performs all AI
  inference.

## Environments

- Host Python: 3.14.4.
- `.venv-viewer`: PySerial and Pillow for the GUI and serial tools.
- `.venv-fold`: PyTorch 2.9.0 CPU, NumPy, PyYAML, and Pillow for model audits.
- No CUDA environment or available GPU has been established.

## Known unresolved reproduction issue

The model-folding pipeline works internally: its Python integer simulator and
rebuilt C engine agree byte-for-byte. The rebuilt blob does not reproduce the
released blob hash, probably because the release used an unpinned PyTorch
nightly/CUDA/WSL environment. No custom UF2 has been built from the rebuilt
artifact. This is recorded as a qualified Lesson 11 result and is independent
of the current AdaLN research.

## Extended capstone Part 1 — AdaLN-single

### Hypothesis

Replace eight block-specific AdaLN MLPs with one shared time-and-class MLP plus
small learned block offsets. The raw modulation representation would fall from
245,760 to 36,864 values, an 85% reduction.

### Completed tests

1. **Post-hoc additive approximation**
   - All five conditions, eight deployed times, eight blocks, and six
     modulation vectors were evaluated from the released `m3_long_cfg` EMA.
   - Overall modulation relative RMSE: 22.5%.
   - Plain final-latent relative error: 45.5%.
   - CFG-4 final-latent relative error: 55.6%.
   - Pixel MAE: 18.52 levels plain; 25.68 levels CFG 4.
   - Approximate images retained faces but visibly lost contrast and detail.

2. **Independent held-out velocity loss**
   - 128 deterministic images from the original final-1,000-image FFHQ
     holdout range; 256 latents after horizontal flips.
   - Assigned-class MSE: original 1.1246, approximation 1.3331 (+18.5%).
   - Null MSE: original 1.1277, approximation 1.3314 (+18.1%).
   - Assigned-class target cosine: 0.6545 to 0.5698.
   - Error increase was worst near the clean end: +29.0% at `t=0.25` and
     +50.8% at `t=0.125`.

### Current conclusion

Deleting or algebraically compressing the released model's AdaLN tables is not
acceptable. However, the conditioning-only trainable student recovered most
of the lost accuracy without allowing the rest of the transformer to adapt.
This changes AdaLN-single from a speculative option into a credible candidate.

No released checkpoint, model blob, or Pico firmware was modified by these
experiments.

### Conditioning-only pilot result

- Architecture: one shared time-and-class conditioning MLP plus per-block
  offsets; all non-conditioning transformer parameters frozen.
- Trainable parameters: 105,216.
- Training data: 640 cached states from 80 balanced teacher trajectories.
- Optimization: 600 steps, batch size 16, CPU, about 104 seconds.
- Held-out MSE: 1.1605 versus 1.1246 for the teacher.
- Held-out penalty: 3.2%, down from 18.5% before training.
- Gap recovered: 82.8%.
- Target cosine: 0.6408 versus 0.6545 for the teacher and 0.5698 before
  training.
- Late timestep penalties: 5.4% at `t=0.25` and 6.6% at `t=0.125`, down from
  29.0% and 50.8%.
- Four-seed plain image comparison: latent error 15.9%, pixel MAE 7.12, PSNR
  27.98 dB.
- Four-seed CFG-4 comparison: latent error 30.3%, pixel MAE 11.91, PSNR 23.50
  dB.

Visual inspection found plain outputs very close to the teacher. CFG magnifies
the residual mismatch, but the sampled faces remain coherent and retain their
class behavior. The experiment is small, so it does not yet establish parity
in diversity or quality across the full distribution.

### Expanded conditioning result

The pool was expanded to 320 trajectories and training to 1,600 steps. States
at `t <= 0.25` were sampled with 2.5x weight.

- Held-out MSE: 1.1550, a 2.7% penalty versus the teacher.
- Target cosine: 0.6428.
- Original approximation gap recovered: 85.4%.
- Late penalties improved to 2.51% at `t=0.25` and 3.26% at `t=0.125`.
- Several middle-timestep penalties worsened, showing an optimization tradeoff.
- On the same 32-seed grid, pixel agreement was slightly worse than the pilot
  in plain, CFG-2, CFG-4, and CFG-6 modes, although images remained coherent.

Conclusion: more conditioning-only training can improve velocity loss, but
late-step oversampling is not sufficient to improve full sampled images.
Checkpoint selection must include complete trajectory and decoded-image
metrics rather than relying on isolated velocity MSE.

## Validation assets now available

- A 128-image research subset from the FFHQ holdout range.
- Corresponding gender/smile annotations.
- 256 cached normalized latents including horizontal flips.
- Dataset-server images are JPEG renditions, not byte-identical copies of the
  original PNG training archive. This is sufficient for a preliminary audit,
  not an exact reproduction of upstream validation loss.

## Next-step options

### Option A — Balanced conditioning-only comparison (recommended next)

Reuse the new 320-trajectory pool but reduce the late-step emphasis, for
example from 2.5x to 1.25–1.5x. Evaluate checkpoints using the matched 32-seed
plain/CFG grid in addition to held-out velocity loss.

- Hardware: CPU is possible; CUDA would be faster.
- Downloads: none required for an initial teacher-distillation pilot.
- Risk: low; it does not change firmware or released artifacts.
- Decision gate: can it retain most of the improved late-step accuracy without
  worsening middle steps or decoded teacher agreement?
- Interpretation: if conditioning-only gains plateau, a limited whole-model
  fine-tune becomes the next escalation rather than the default next step.

### Option B — Whole-model shared-AdaLN fine-tuning

After initializing from Option A, unfreeze the transformer and optimize a
mixture of true flow loss and frozen-teacher velocity distillation.

- Hardware: CUDA GPU strongly recommended.
- Data: ideally the original 2.11 GB FFHQ archive and approximately 0.6 GB of
  encoded/augmented latents, plus checkpoints and evaluation outputs.
- Benefit: highest chance of restoring quality while retaining the smaller
  architecture.
- Cost: environment setup, hours of training, broader quality evaluation, and
  possible tuning of loss weights and learning rates.

### Option C — Quantization-damage map

Pause AdaLN training and begin Extended Capstone Part 2. Compare the float and
fake-int8 graph at every quantization boundary across representative sampling
trajectories. Rank layers by clipping, outliers, quantization error, and effect
on final velocity.

- Hardware: CPU is adequate for a bounded audit.
- Downloads: no full dataset required initially.
- Benefit: identifies whether ReLU-squared, AdaLN amplification, attention, or
  a small number of layers dominate the remaining float-to-int8 quality gap.
- This work is useful even if AdaLN-single is abandoned.

### Option D — Full retraining from scratch

Train a new shared-AdaLN model directly from FFHQ rather than adapting the
released teacher.

- Hardware/data cost: highest; a strong CUDA GPU and full dataset pipeline are
  effectively required.
- Benefit: maximum architectural freedom.
- Risk: unnecessary until fine-tuning has been shown insufficient.
- Recommendation: do not choose this first.

### Option E — Package and pause research

Treat the validated Pico firmware and GUI as the finished practical project,
preserve the research results, and postpone model training until suitable GPU
hardware is available.

## Work that should wait

Do not redesign the blob format, integer folding, C engine, or Pico firmware
for AdaLN-single until a trained float student passes held-out loss and image
quality gates. Expanding the current approximation into firmware would only
make a known-bad model harder to debug.

## Recommended continuation

Proceed with **Option A** for one controlled, balanced-weight comparison using
the already cached teacher states. If the velocity/image tradeoff persists,
stop tuning conditioning alone and move to a limited whole-body fine-tune.

## Extended capstone Part 2 — First quantization damage map

A desktop fake-deployment audit now covers 20 plain and 20 CFG-4 trajectories,
including both CFG calls, for 480 DiT evaluations. Float versus fake-int
velocity MSE was 0.005049, relative RMSE 8.65%, and cosine 0.9967.

Block-level oracle repairs identified the leading causal sources:

- Block 4 attention: 26.26% final-MSE recovery.
- Block 5 attention: 20.81%.
- Block 7 MLP: 10.37%.
- Block 1 attention: 8.81%.

Within Blocks 4 and 5, repairing the quantized AdaLN-normalized attention
inputs recovered 19.18% and 14.08%. Repairing their softmax grids recovered
5.00% and 4.19%. Weight repairs and most other individual Q/K/V boundaries
were small.

The largest local discrepancy, Block 5 pre-normalization Q at 39.7% relative
RMSE, recovered only 1.51% when repaired because later Q normalization removes
most of it. This validates the need for causal repairs in addition to local
error rankings.

Block 4/5 input clipping was concentrated at `t=1.0` and `t=0.875`, around
0.8%, and was similar for plain, conditional, and null calls. The next
recommended quantization experiment is a narrow scale sweep for these two
attention-input grids, scored by local clipping, final velocity error, and
complete sampling trajectories. Do not begin QAT or firmware changes until
that cheaper scale test establishes the direction of improvement.

### Attention-input scale sweep result

The sweep passed its fake-quant quality gate. Best tested multipliers were
1.80x for Block 4 and 1.40x for Block 5. They nearly eliminated attention-input
clipping and reduced teacher-state velocity MSE by 26.4%.

Across 32 complete trajectories, plain pixel MAE fell from 5.78 to 3.89 and
CFG-4 MAE from 9.39 to 7.32. Corresponding PSNR rose from 29.85 to 32.95 dB
plain and 25.61 to 27.74 dB CFG-4. The improvement is therefore not confined
to isolated velocity calls.

The next gate was to modify the available calibration values, fold a new blob,
and test with the byte-exact integer simulator before considering a Pico flash.

### Exact-fold gate passed

Inspection corrected the preceding assumption: this released calibration has
scalar `att_in` entries but no `att_in.__pc` entries. A derived calibration
changed only the two existing scalar values and preserved the source file.

The tuned fold retained the same 2,567,828-byte size and byte-identical model
header. Exact integer comparison against float improved plain pixel MAE from
3.08 to 2.33 and CFG-4 MAE from 6.75 to 5.97 on four fixed seeds/classes.
The portable C engine matched all four tuned Python goldens byte-for-byte,
covering the repository's CFG-4, CFG-6, CFG-8, and null/plain cases.

The tuned blob is ready for a broader exact-simulator seed gate or a custom
firmware build. It has not been flashed. Because the original release fold is
not reproducible in the current unpinned environment, all causal comparisons
use the same-environment rebuilt baseline rather than claiming release-hash
equivalence.

### Custom firmware gate passed

The tuned blob was built into a Pico 2 UF2 with SDK 2.2.0 and ARM GCC 14.2.1.
Picotool verified the flash write and rebooted the board. Seed 42, class 1,
four steps, CFG 4 generated on the physical RP2350 in 5,344 ms with CRC32
`94ee4f2c`. The full 49,152-byte RGB frame matches the tuned Python integer
simulator byte-for-byte.

The matching plain request completed in 3,401 ms with CRC32 `b693c574` and was
also byte-exact, covering the single-pass path alongside CFG's dual-pass path.

The custom firmware is now running on the board. Its named UF2 is
`device-output/capstone-02-exact-fold/pico_faces_m3_long_cfg_attn45.uf2`.
The original released UF2 remains in `upstream/pico-faces/uf2/` for rollback.
