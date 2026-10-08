# Lesson 08 — Conditioning and classifier-free guidance

> This lesson was split after review because it introduced too many mechanisms
> at once. Complete [Lesson 08A](08a-classes-time-adaln.md) first, then
> [Lesson 08B](08b-classifier-free-guidance.md). The original combined notes
> remain below as a later reference.

## Outcome

Understand how one DiT changes its prediction for the current time and requested
face class, and why classifier-free guidance requires two DiT evaluations per
Euler step.

## Why the DiT needs time

The current latent alone does not tell the complete story. Early in generation,
the state is mostly noise and the model should establish broad structure. Late
in generation, it should make smaller, detail-oriented corrections.

The DiT therefore receives the current time `t` along with the latent:

```text
velocity = DiT(current_latent, time, class)
```

This `time` is the location along the noise-to-image journey. It is distinct
from `dt`, the amount the sampler advances after receiving a velocity.

## Why the DiT needs a class

The released model recognizes five conditioning values:

| Class | Meaning |
| ---: | --- |
| 0 | female, neutral |
| 1 | female, smiling |
| 2 | male, neutral |
| 3 | male, smiling |
| 4 | unconditional/null |

The labels statistically steer the generated result; they are not strict rules
about any individual image. The unconditional class means “predict without one
of the four requested attribute combinations.”

## How AdaLN applies the conditions

Transformer blocks repeatedly normalize their activations so values stay on a
manageable scale. **Adaptive Layer Normalization**, or **AdaLN**, lets the time
and class condition modify what happens around that normalization.

A simplified view is:

```text
normalized = Normalize(token)
conditioned = scale(time, class) * normalized + shift(time, class)
```

The actual block has several condition-dependent values and residual branches,
but the central idea is that time and class adjust the same shared DiT rather
than selecting an entirely different model.

**AdaLN-Zero** initializes the conditioned residual contribution at or near
zero, letting each block begin close to an identity operation and learn useful
changes gradually during training.

The training model can calculate conditioning values with a small network. On
the Pico, only eight timesteps and five classes are needed. The project
precomputes the required values and stores lookup tables indexed by timestep,
class, and layer. This trades flash space for less runtime computation.

## Conditional and unconditional predictions

For classifier-free guidance, or **CFG**, the same current latent and time are
sent through the DiT twice:

```text
v_null = DiT(latent, time, unconditional class)
v_cond = DiT(latent, time, requested class)
```

The difference isolates the effect of asking for the class:

```text
class direction = v_cond - v_null
```

Guidance amplifies that difference:

```text
v_guided = v_null + w * (v_cond - v_null)
```

Important: `w` multiplies only the class-specific difference, not the entire
conditional velocity.

## One-number example

Suppose:

```text
v_null = 2
v_cond = 3
w = 4
```

Then:

```text
class direction = 3 - 2 = 1
v_guided = 2 + 4*(1) = 6
```

Useful landmarks:

- `w = 0` gives the unconditional prediction.
- `w = 1` gives the ordinary conditional prediction.
- `w > 1` pushes beyond the conditional prediction in the class direction.

The released firmware has baked guidance settings 4, 6, and 8. Stronger
guidance can enforce the requested statistics more strongly, but excessive
guidance can reduce variety or create exaggerated artifacts.

## Why guidance costs roughly twice as much

Every Euler step now performs two full DiT evaluations—one null and one
conditional—before combining them. The VAE decoder still runs only once after
the final latent.

```text
plain sampling:   1 DiT pass per Euler step
guided sampling:  2 DiT passes per Euler step
```

At eight Euler steps, that means eight DiT passes without CFG versus sixteen
with CFG.

## Run the CFG calculator

```powershell
python .\lesson_08_cfg.py
```

The script applies the formula elementwise to tiny velocity vectors for
guidance weights 0, 1, 4, 6, and 8.

## Checkpoint

1. Why does the DiT need the current time as well as the current latent?
2. What is the difference between AdaLN conditioning and CFG?
3. If `v_null = 2`, `v_cond = 3`, and `w = 4`, what is `v_guided`?
4. Why does CFG approximately double the DiT work per Euler step?

