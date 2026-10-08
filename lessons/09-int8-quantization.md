# Lesson 09 — Int8 quantization: storing approximate numbers cheaply

## Goal

Understand how the trained model's ordinary real-valued numbers can be
represented by small integers, measure the error this introduces, and see why
calibration and quantization-aware training matter on the Pico.

## 1. The central tradeoff

A typical floating-point model stores each value in 32 bits. An `int8` value
uses 8 bits, so it needs one quarter of the storage. Integer operations are
also a much better fit for the RP2350's fast DSP instructions.

The cost is that an integer cannot represent every real number. Quantization
replaces a continuous range of values with a finite grid of allowed values.

For a simple *symmetric* quantizer:

```text
q             = clamp(round(x / scale), -127, 127)
reconstructed = q * scale
```

- `x` is the original real value.
- `q` is its stored integer code.
- `scale` is the real-world spacing between neighboring integer codes.
- `clamp` prevents the code from exceeding the available range.

The deployed Pico code uses the symmetric range `-127..127`; it deliberately
does not use `-128`.

## 2. One value by hand

Suppose `scale = 0.1` and `x = 0.26`:

```text
q             = round(0.26 / 0.1) = round(2.6) = 3
reconstructed = 3 * 0.1 = 0.3
error         = 0.3 - 0.26 = 0.04
```

The integer `3` is not the value itself. It means “three grid spacings from
zero.” The scale is needed to interpret it.

## 3. Why the scale matters

A smaller scale gives closely spaced representable values, so it preserves
small differences. But its total range is also smaller:

```text
largest representable magnitude = 127 * scale
```

Values outside that range are clipped. A larger scale covers a wider range
but makes the grid coarser. Quantization is therefore a balance between:

- **rounding error** from snapping values to a coarse grid; and
- **clipping error** from values falling outside the grid.

For a first approximation, a scale that covers a tensor's largest magnitude
is:

```text
scale = max(abs(values)) / 127
```

Real quantization systems may choose a different threshold when accepting a
little clipping reduces the overall error.

## 4. Run the experiment

From the `pico-faces-lab` directory, run:

```powershell
python .\lesson_09_int8_quantization.py
```

The script performs three small experiments:

1. It quantizes a vector, reconstructs it, and reports each error.
2. It compares several scales to expose the range-versus-resolution tradeoff.
3. It compares one scale for a whole matrix with one scale per output channel.

The last comparison matters because different channels can have very different
ranges. One global scale may be set by a large-valued channel and erase most of
the detail in a small-valued channel. Per-channel scales preserve that detail,
at the cost of storing and applying more scales.

## 5. Weights and activations are different calibration problems

**Weights** are fixed after training, so their ranges can be inspected
directly.

**Activations** are temporary values produced by a particular input as it
moves through the network. Their ranges can change with the random latent,
timestep, class, and CFG path. We therefore run representative sampling
trajectories and record the observed activation ranges. That process is called
**calibration**.

For CFG, calibration must cover both the unconditional and conditional DiT
calls. Otherwise a scale that works for one path may clip values in the other.
The upstream project calibrates using complete sampling trajectories rather
than only clean images or isolated random tensors.

## 6. Quantization-aware training

Converting an already-trained float model directly is called
**post-training quantization**. It is simple, but each rounding error can alter
later activations.

In **quantization-aware training** (QAT), training simulates those rounding and
clipping effects while the adjustable weights are still being refined. The
model can then learn values that behave better on the available integer grid.
The upstream project uses a frozen float model as a teacher while refining the
quantized student.

## 7. Integer multiplication still needs a wider accumulator

Even when two inputs fit in `int8`, their product does not necessarily fit:

```text
127 * 127 = 16129
```

A matrix multiplication also adds many such products. The implementation
therefore accumulates them in a wider integer, then *requantizes* the result
back onto the next layer's int8 grid. Requantization applies a multiplier and
shift, rounds, and saturates the result to `-127..127`.

Conceptually:

```text
int8 inputs -> wide products and sum -> rescale -> round -> saturate -> int8
```

The exact deployed rounding and saturation rules matter: the desktop integer
simulator and the Pico implementation must make the same decision for every
value.

## Checkpoint

Answer these in your own words before Lesson 10:

1. With `x = 0.26` and `scale = 0.1`, what integer is stored, what value is
   reconstructed, and what is the error?
2. Why can a scale that is too small have *more* error even though its grid is
   finer?
3. Why can per-channel scales preserve a small-valued channel better than one
   scale for the entire tensor?
4. Why must activation calibration use representative diffusion sampling
   trajectories, including both CFG paths?
5. Why are int8 products accumulated in a wider integer before being converted
   back to int8?

Do not mark the lesson complete yet. We will review the script output and your
checkpoint answers together.
