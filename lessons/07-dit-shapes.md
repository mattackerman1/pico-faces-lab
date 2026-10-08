# Lesson 07 — Through the Diffusion Transformer

## Outcome

Trace every tensor shape from the 16 x 16 x 8 latent to 64 transformer tokens
and back to a 16 x 16 x 8 velocity prediction.

This lesson is about data organization. We will postpone the detailed attention
calculations.

## Step 1: Start with the latent

At one Euler step, the DiT receives the current latent:

```text
16 high x 16 wide x 8 channels = 2,048 values
```

It also receives the current time and class condition. Those conditions affect
the processing, but they do not change the main tensor's shape.

## Step 2: Patchify

Divide the 16 x 16 spatial grid into non-overlapping 2 x 2 patches. Every patch
includes all eight channels:

```text
one patch = 2 x 2 x 8 = 32 values
```

There are eight patches across and eight down:

```text
16 / 2 = 8
8 x 8 = 64 patches
```

After flattening each patch, the shape is:

```text
64 patches x 32 values
```

This rearrangement has not discarded anything:

```text
64 x 32 = 2,048 values
```

## Step 3: Embed each patch as a token

A learned linear layer transforms every 32-value patch into a 128-value token:

```text
64 x 32  --embedding-->  64 x 128
```

This is not another compression step. It expands each patch into a richer
working representation so the transformer has room to represent useful
features and relationships.

The number 128 is the model's **width**, an architecture choice rather than a
quantity forced by the 32 inputs. The embedding is a learned transformation
from 32 numbers to 128 working features. A wider model can represent more
patterns but needs more weights, multiplication, activation memory, and time.
A narrower model is cheaper but may lack capacity. The author selected 128 as
part of the quality-versus-device-budget tradeoff; the article does not claim
that 128 is uniquely correct.

The patch embedding initially represents patch **content**. A separate
sinusoidal positional embedding is then added to tell it where the patch came
from. Without position information, the transformer would know which patch
values exist but not where each patch belongs in the 8 x 8 layout.

## Step 4: Process the tokens

Every DiT block receives and returns the same `64 x 128` shape. The released
fast model uses eight blocks; the quality model uses twelve.

A simplified block contains two important transformations:

1. **Self-attention:** each patch token can gather relevant information from
   other patch tokens. A patch near one eye can therefore be influenced by the
   other eye, face outline, or mouth region.
2. **Dense/MLP section:** independently transforms the 128 working values in
   each token. Pico-Faces uses a squared-ReLU activation whose sparsity can be
   exploited by the inference engine.

Squared-ReLU means:

```text
ReLU-squared(x) = max(0, x)^2

x = -2.0  -> 0
x = -0.5  -> 0
x =  0.5  -> 0.25
x =  2.0  -> 4
```

All negative inputs become exact zeros. A collection in which many values are
zero is called **sparse**. Compared with smooth activations that leave many
small nonzero values, ReLU-squared creates more opportunities for exact zeros;
quantization can turn additional tiny values into zero. The Pico inference
engine exploits this sparsity to avoid some useless arithmetic. Squaring also
emphasizes larger positive activations, so this function changes both sparsity
and the shape of the positive response.

### What is an activation?

A neural-network layer first combines input values with learned weights and a
bias. The result before the nonlinear function is often called a
**pre-activation**. Applying a function such as ReLU-squared produces the
**activation**:

```text
input values
    |
    v
weighted sum + bias  -> pre-activation z
    |
    v
ReLU-squared(z)      -> activation a
    |
    v
input to the next layer
```

Weights and activations are different:

- **Weights** are learned during training and then stored as model parameters.
  For inference, they remain fixed.
- **Activations** are temporary intermediate values calculated for the current
  input. They change for every token, layer, timestep, and generated image.

Zeros are not removed from the activation tensor or treated as nonexistent.
They remain valid values. An optimized operation can skip a multiplication when
it knows that multiplying by a zero activation cannot contribute to the sum.
This is how activation sparsity can save computation.

Residual connections add each section's result back to its input. Time and
class conditioning alter normalization inside the block; Lesson 08 will cover
that mechanism.

The shape remains:

```text
64 tokens x 128 values
```

## Step 5: Project back to velocity patches

The final projection converts each 128-value token into 32 predicted velocity
values:

```text
64 x 128  --output projection-->  64 x 32
```

These are not the original input patches. They are predicted velocity patches.

## Step 6: Unpatchify

Put the 64 velocity patches back into their 8 x 8 spatial arrangement. Each
patch fills a 2 x 2 region with eight channels:

```text
64 x 32  --unpatchify-->  16 x 16 x 8
```

The output now has the same shape as the current latent, so the Euler sampler
can perform its elementwise update:

```text
next_latent = current_latent + dt * predicted_velocity
```

That update happens **outside** the DiT:

```text
DiT:      current latent -> predicted velocity
sampler:  current latent + dt * predicted velocity -> next latent
```

The next latent is then passed into the DiT on the following Euler step. Thus
the DiT is the direction predictor and the sampler is the state updater.

## Why attention operates on patches

With 64 tokens, one attention head considers a 64 x 64 set of token-to-token
relationships: 4,096 entries. Treating all 256 latent positions as separate
tokens would require 256 x 256 = 65,536 relationships per head. The 2 x 2
patches reduce that work by a factor of 16.

## Run the shape tracer

```powershell
python .\lesson_07_dit_shapes.py
```

The script creates a numbered synthetic latent, patchifies it, and unpatchifies
it again. It asserts that every value returns to its original location. It then
prints the shapes used by the learned embedding, DiT blocks, and output layer.

## Checkpoint

1. Does patchifying from `16 x 16 x 8` to `64 x 32` discard any values? How can
   you tell?
2. Why does the model add positional information to the tokens?
3. Why must the final projection and unpatchifying stages return to
   `16 x 16 x 8`?

