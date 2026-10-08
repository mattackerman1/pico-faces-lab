# Extended capstone — Improve the deployed model

The GUI capstone is working. The extended capstone investigates architectural
and training changes that could make the model smaller or more accurate on the
Pico 2. Each part begins with a low-cost feasibility audit before committing to
training or firmware changes.

## Part 1 — AdaLN-single conditioning compression

### Question

Can the model replace its block-specific AdaLN conditioning functions with one
shared time-and-class function plus learned block offsets, without materially
damaging velocity predictions or generated images?

### Why it matters

The current exporter precomputes AdaLN for five conditions, eight timesteps,
and every transformer block. In the fast depth-8 model, the raw block
modulation count is:

```text
5 conditions × 8 times × 8 blocks × 6 vectors × 128 values
= 245,760 values
```

An AdaLN-single-inspired representation would use one shared time-and-class
result plus eight block embeddings:

```text
(5 × 8 × 6 × 128) + (8 × 6 × 128)
= 36,864 values
```

This is an 85% reduction in raw modulation values. Real blob savings will
depend on how shared modulation is combined with the block-specific integer
scales and requantization multipliers currently produced by folding.

### Important adaptation

PixArt-alpha's original AdaLN-single shares time conditioning while text enters
through cross-attention. Pico Faces has no cross-attention; its classes enter
through AdaLN. Our candidate must therefore share **time plus class**, not time
alone:

```text
S_block(time, class) = S_shared(time, class) + E_block
```

### Feasibility experiment

1. Load the released floating-point checkpoint.
2. Evaluate every block's six AdaLN vectors at all 40 time/class combinations.
3. Fit the best additive approximation `S_shared(time,class) + E_block`.
4. Report error separately for attention scale/shift/gate and MLP
   scale/shift/gate.
5. Substitute the approximation into desktop floating-point inference.
6. Compare velocities and images over a fixed set of seeds, classes, step
   counts, and CFG settings.
7. Proceed to fine-tuning and a compact integer table format only if the
   approximation is promising.

### Success criteria

- A substantial reduction in conditioning representation size.
- Small velocity error across full sampling trajectories, not just isolated
  dataset latents.
- No obvious loss of face coherence, class control, or diversity.
- A credible integer representation whose runtime arithmetic costs less than
  the flash and quality benefits it provides.

### Status

Initial no-retraining feasibility audit completed on the released
`m3_long_cfg` EMA checkpoint.

The best least-squares additive representation reduced the raw modulation
count from 245,760 to 36,864 values (85%), but it was not accurate enough to
substitute directly into the existing model:

| Measurement | Plain | CFG 4 |
| --- | ---: | ---: |
| Final latent relative error | 45.5% | 55.6% |
| Mean absolute RGB error | 18.52 levels | 25.68 levels |
| PSNR | 20.75 dB | 17.46 dB |

Across all modulation values, relative RMSE was 22.5% and cosine similarity
was 0.9744. MLP shift was the least compressible component at 29.8% relative
RMSE. During sampling, velocity error accumulated sharply in the late steps;
mean velocity cosine fell to about 0.22 at the final timestep in both modes.

The approximated model retained recognizable face structure and some class
behavior, but produced visibly washed-out, altered images with substantial
loss of detail. CFG amplified the discrepancy.

**Decision:** AdaLN-single remains plausible only with architectural
fine-tuning or retraining. It cannot be applied to the released checkpoint as
a lossless or acceptable post-export compression. The next stage, if pursued,
is a teacher-distilled shared-AdaLN student—not firmware table deletion.

Reproduce with `capstone_01_adaln_single.py`. Full numeric output and comparison
images are under `device-output/capstone-01-adaln-single/`.

### Held-out velocity-loss audit

To test correctness independently of the original model's output, 128
deterministically selected FFHQ images were fetched from the original
last-1,000-image holdout range. Including the same horizontal-flip augmentation
used upstream produced 256 validation latents. Each latent was mixed with known
noise at one of the eight deployed timesteps, giving a known flow-matching
target velocity `noise - clean_latent`.

| Evaluation | Original MSE | Shared-AdaLN MSE | Increase |
| --- | ---: | ---: | ---: |
| Assigned class | 1.1246 | 1.3331 | 18.5% |
| Null condition | 1.1277 | 1.3314 | 18.1% |

For assigned classes, mean velocity cosine against the correct target fell
from 0.6545 to 0.5698. The loss increase was similar across face classes. It
was strongly timestep-dependent: only 3.2% at `t=0.625`, but 29.0% at
`t=0.25` and 50.8% at `t=0.125`, the final prediction point nearest the clean
latent. This independently supports the visual observation that fine detail
and coherence are damaged.

This is a bounded preliminary validation subset. Its indices are genuinely
held out, but images were obtained from the Hugging Face dataset-server JPEG
cache instead of the byte-identical PNG archive used in upstream training.
Reproduce with `capstone_01_fetch_validation.py` followed by
`capstone_01_velocity_loss.py`; results are in
`device-output/capstone-01-adaln-single/validation/`.

### Conditioning-only student pilot

The next experiment replaced the eight block-specific conditioning MLPs with
one trainable shared time-and-class MLP plus learned block offsets. All
attention, token MLP, embedding, and output weights were copied from the
teacher and frozen. Only 105,216 conditioning parameters were trained, using
640 cached states from 80 balanced teacher sampling trajectories.

After 600 CPU optimization steps, the independently held-out assigned-class
velocity results were:

| Model | Target MSE | Target cosine |
| --- | ---: | ---: |
| Original teacher | 1.1246 | 0.6545 |
| Untrained shared approximation | 1.3331 | 0.5698 |
| Trained conditioning-only student | 1.1605 | 0.6408 |

The shared student reduced the held-out penalty from 18.5% to 3.2%, recovering
82.8% of the gap to the teacher without changing the transformer body. The
largest remaining penalties were at the final two prediction times: 5.4% at
`t=0.25` and 6.6% at `t=0.125`, down from 29.0% and 50.8%.

Image comparisons over four fixed class/seed pairs also improved strongly:

| Measurement | Plain | CFG 4 |
| --- | ---: | ---: |
| Final latent relative error | 15.9% | 30.3% |
| Mean absolute RGB error | 7.12 levels | 11.91 levels |
| PSNR | 27.98 dB | 23.50 dB |

The plain outputs are visually very close to the teacher. CFG amplifies the
remaining conditioning mismatch, but the tested student outputs remain
coherent faces with the intended class behavior. This small pilot is evidence
that shared conditioning can work; it is not yet proof of equal quality or
diversity over the full input distribution.

**Updated decision:** continue improving and evaluating the conditioning-only
student before unfreezing the transformer body. The best next experiment is a
larger, balanced teacher-state pool with extra emphasis on late timesteps,
followed by a broader fixed-seed image/class/CFG evaluation. Exporter,
quantization, and firmware work should still wait for that quality gate.

Reproduce training with `capstone_01_train_student.py` and images with
`capstone_01_student_images.py`. Results and the conditioning checkpoint are
under `device-output/capstone-01-adaln-single/student-pilot/`.

### Expanded, late-weighted student experiment

A second conditioning-only student used 320 balanced teacher trajectories
(2,560 cached states), 1,600 optimization steps, and 2.5x sampling weight for
the final two prediction times. The same transformer body remained frozen.

The larger run improved aggregate held-out velocity loss modestly:

| Held-out result | 80-trajectory pilot | Expanded run | Teacher |
| --- | ---: | ---: | ---: |
| Velocity MSE | 1.1605 | **1.1550** | 1.1246 |
| MSE penalty | 3.2% | **2.7%** | — |
| Target cosine | 0.6408 | **0.6428** | 0.6545 |

Late weighting achieved its local goal. Penalties at `t=0.25` and `t=0.125`
fell from 5.37% and 6.58% to 2.51% and 3.26%. However, penalties at several
middle timesteps rose, including `t=0.75` from 2.59% to 4.09% and `t=0.625`
from 1.75% to 3.37%.

A matched evaluation used 32 seeds (eight per class) for both checkpoints:

| Mode | Pilot MAE | Expanded MAE | Pilot PSNR | Expanded PSNR |
| --- | ---: | ---: | ---: | ---: |
| Plain | **7.11** | 7.58 | **27.97 dB** | 27.46 dB |
| CFG 2 | **8.93** | 9.16 | **25.93 dB** | 25.80 dB |
| CFG 4 | **12.63** | 12.86 | **23.09 dB** | 22.87 dB |
| CFG 6 | **16.40** | 16.94 | **20.93 dB** | 20.50 dB |

The expanded model slightly reduced final-latent drift for CFG 4 and CFG 6,
but its decoded pixels were not closer to the teacher in any tested mode.
Visual inspection still found coherent faces and preserved class behavior.

**Interpretation:** additional conditioning-only training has not plateaued on
velocity loss, but strongly emphasizing the final timesteps shifts error into
earlier trajectory stages. Since sampling compounds all eight predictions,
better late-step scores alone do not guarantee closer final images. The next
conditioning experiment should use a milder or loss-balanced timestep
objective and select checkpoints using full-trajectory image metrics, not
isolated velocity MSE alone. If that tradeoff persists, limited whole-body
fine-tuning is more promising than further increasing late-step weight.

Reproduce with `capstone_01_train_student.py` using 320 trajectories, 1,600
steps, and late weight 2.5, then run `capstone_01_student_grid.py`. Results are under
`device-output/capstone-01-adaln-single/student-expanded/`.

## Part 2 — Quantization robustness

### First DiT damage map

The first desktop audit ran 20 plain and 20 CFG-4 teacher trajectories through
the released floating-point checkpoint and its fake-deployment int8 graph.
Both conditional and null calls were measured, for 480 DiT evaluations total.
It recorded local grid rounding/clipping, cumulative residual divergence, and
block-level oracle repairs. An oracle repair leaves the complete graph
quantized except for one component; the reduction in final velocity MSE is a
causal importance estimate.

Overall fake-int versus float velocity error was MSE 0.005049, relative RMSE
8.65%, and cosine 0.9967. The largest block-level repair effects were:

| Component repaired | Final velocity MSE recovered |
| --- | ---: |
| Block 4 attention | 26.26% |
| Block 5 attention | 20.81% |
| Block 7 MLP | 10.37% |
| Block 1 attention | 8.81% |
| Block 6 attention | 7.14% |

Individual repairs inside the two leading attention blocks localized most of
their damage to the quantized AdaLN-normalized attention inputs: 19.18% MSE
recovery for Block 4 and 14.08% for Block 5. Their fixed-grid softmax outputs
were secondary at 5.00% and 4.19%. Weight-only repairs were small.

Block 5's pre-normalization query grid had the largest local discrepancy
(39.7% relative RMSE, cosine 0.922), but repairing it recovered only 1.51% of
final MSE. Subsequent Q normalization removes most of that error. This is the
clearest example of why local error alone cannot rank quantization priorities.

The attention-input problem is timestep-specific. At `t=1.0` and `t=0.875`,
Blocks 4 and 5 clipped about 0.8% of input values and showed large local error;
clipping was nearly absent from `t=0.625` onward. Plain, conditional, and null
paths behaved similarly, so the leading issue is not a CFG-only calibration
failure.

**Next candidate experiment:** selectively adjust the Block 4/5 attention
input scales and measure the clipping-versus-rounding tradeoff with the same
oracle and full-trajectory metrics. A small scale sweep should precede QAT or
wider-precision changes.

This is a first-pass fake-quant audit, not the byte-exact integer simulator,
and it covers the DiT rather than the VAE decoder. Full results are under
`device-output/capstone-02-quant-damage/`; reproduce with
`capstone_02_quant_damage.py` and render with `capstone_02_plot.py`.

### Block 4/5 attention-input scale sweep

The proposed scale test was completed without retraining. A 36-point coarse
sweep was followed by a 25-point refinement. The best tested fake-deployment
setting multiplied Block 4's calibrated attention-input scale by 1.80 and
Block 5's by 1.40.

| Teacher-state velocity metric | Original scales | Rescaled | Change |
| --- | ---: | ---: | ---: |
| MSE | 0.005049 | 0.003718 | -26.4% |
| Relative RMSE | 8.65% | 7.43% | -14.1% |
| Cosine | 0.9967 | 0.9972 | improved |
| Block 4 input clipping | 0.284% | 0.000% | eliminated in audit |
| Block 5 input clipping | 0.275% | 0.0016% | nearly eliminated |

The benefit survived 32 complete eight-step trajectories rather than merely
improving isolated teacher states:

| Mode | Metric | Original fake-int | Rescaled fake-int |
| --- | --- | ---: | ---: |
| Plain | Final latent relative error | 13.40% | **9.65%** |
| Plain | Pixel MAE | 5.78 | **3.89** |
| Plain | PSNR | 29.85 dB | **32.95 dB** |
| CFG 4 | Final latent relative error | 22.83% | **18.93%** |
| CFG 4 | Pixel MAE | 9.39 | **7.32** |
| CFG 4 | PSNR | 25.61 dB | **27.74 dB** |

The refined curve established the clipping/rounding tradeoff. Increasing
Block 4 beyond 1.80 and Block 5 beyond roughly 1.40 removed no meaningful
additional clipping and began paying for coarser grid resolution. Visual grids
retained coherent faces and class behavior while more closely matching float.

**Decision:** the scale change passes the fake-quant feasibility gate. Before
firmware use, change the released calibration entries that actually exist,
fold a new model blob, and repeat the comparison in the byte-exact integer
simulator. This is necessary because fake quantization does not reproduce every
integer RMSNorm, softmax, requantization, and saturation detail. The released
calibration has scalar `att_in` entries; it does not contain optional
`att_in.__pc` per-channel entries.

Reproduce with `capstone_02_scale_sweep.py`. Coarse and refined results are in
`device-output/capstone-02-scale-sweep/` and
`device-output/capstone-02-scale-refined/`.

### Exact fold and integer verification

A derived calibration was created without modifying the released file. Only
`att_in.4` and `att_in.5` changed, from 7.5036 to 13.5064 (1.80x) and 6.2364
to 8.7309 (1.40x). Folding produced a separate 2,567,828-byte blob with the
same geometry and a byte-identical `rf_cfg.h` header. It differs from the
same-environment baseline blob in 41,621 bytes because the scales affect
folded per-condition/per-timestep normalization and requantization tables.

On four fixed seeds/classes at four Euler steps, byte-exact integer simulation
retained the fake-quant improvement:

| Mode | Metric versus float | Baseline exact integer | Tuned exact integer |
| --- | --- | ---: | ---: |
| Plain | Pixel MAE | 3.08 | **2.33** |
| Plain | PSNR | 35.73 dB | **37.95 dB** |
| CFG 4 | Pixel MAE | 6.75 | **5.97** |
| CFG 4 | PSNR | 28.89 dB | **29.65 dB** |

The existing portable C engine then loaded the tuned blob and generated four
four-step frames. All matched the Python integer-simulator RGB files
byte-for-byte. Their CRC32 values were `785eaf0a`, `3d15e0bf`, `9532f998`,
and `d5c19408`; the cases exercise CFG 4, CFG 6, CFG 8, and the null/plain
path according to the repository's golden-case convention.

This comparison uses the current-environment rebuilt baseline, not the released
blob, because the previously recorded unpinned-PyTorch provenance issue still
prevents reproducing the release hash. Both baseline and tuned blobs here were
folded under the same environment, so the scale change is isolated.

**Decision:** the rescaled blob passes Python exact-integer and portable-C
verification. The experiment still covers only a small seed set and the blob
has not been embedded in or flashed as firmware. Results are under
`device-output/capstone-02-exact-fold/`; reproduce with
`capstone_02_fold_verify.py` followed by the portable `rf_golden` harness.

### Custom Pico 2 firmware and board verification

The verified tuned blob was embedded into a custom RP2350 ARM-S firmware using
Pico SDK 2.2.0 and GCC 14.2.1. Inspection of the raw firmware binary found the
complete tuned blob exactly once at byte offset 51,108. The resulting UF2 is
5,275,648 bytes with SHA-256
`22716999BEA9771210007334C0501F8B8BD51F277F95ED4A0525022CD398084D`.

Picotool placed the connected Pico 2 into BOOTSEL through its USB reset
interface, wrote the UF2, verified the complete flash contents, and rebooted
the application. It re-enumerated as `COM3`.

The physical-board gate used seed 42, four Euler steps, class 1, and CFG 4:

| Board measurement | Result |
| --- | --- |
| Generation time | 5,344 ms |
| Device CRC32 | `94ee4f2c` |
| Host CRC32 | `94ee4f2c` |
| Bytes returned | 49,152 |
| Match to tuned exact simulator | **byte-for-byte** |

The board output is therefore not merely a plausible image: its complete RGB
frame matches the tuned Python exact-integer simulation. The custom firmware
is currently running on the Pico 2. The released UF2 remains available at
`upstream/pico-faces/uf2/pico_faces_m3_long_cfg.uf2` for rollback.

A complementary plain test with the same seed, class, and four steps completed
in 3,401 ms with CRC32 `b693c574` and also matched the tuned simulator
byte-for-byte. Both the single-pass and CFG dual-pass physical-board paths are
therefore verified.

The named custom artifact and full reports are under
`device-output/capstone-02-exact-fold/`.
