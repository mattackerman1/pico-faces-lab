# Lesson 08A — Classes, time, and AdaLN-Zero

## Outcome

Understand what a class is, why time matters, why the model normalizes its
working values, and what the “Zero” in AdaLN-Zero actually does.

## 1. A class is a control label

In machine learning, a **class** is a named category represented by a small
integer. In this project, the class is an input chosen before generation. The
model is not trying to guess which class an existing face belongs to.

The training data includes labels derived from facial-attribute annotations:

| Value | Training label / generation control |
| ---: | --- |
| 0 | female, neutral |
| 1 | female, smiling |
| 2 | male, neutral |
| 3 | male, smiling |
| 4 | null/unconditional |

The four visible categories are the four combinations of two binary dataset
annotations: annotated gender and annotated smile. They were a design choice by
the project author, not categories required by diffusion models. A different
training project could use clothing types, handwritten digits, animal species,
or no classes at all.

These face labels are imperfect statistical annotations, not authoritative
statements about a person's identity. During generation they steer tendencies
in the output rather than guaranteeing a result.

The fifth value is special. The **null class** means “make a velocity prediction
without a requested category.” During training, the code deliberately replaces
the real label with this null value for some examples. This teaches the same
DiT both conditional and unconditional behavior, which Lesson 08B needs.

So the four controls come directly from annotated categories. The null control
is not merely a fifth visible category in the same sense: it is deliberately
taught through class-label dropout, and the upstream data preparation also maps
“no face detected” annotations to the null value.

## 2. The DiT learns a time-dependent function

The DiT is best written as:

```text
predicted velocity = DiT(current latent x, current time t, class y)
```

Time is not merely a loop counter. It tells the model where it is on the
noise-to-data journey:

- Near `t = 0`, the latent is mostly noise; useful predictions establish broad
  structure.
- Near `t = 1`, recognizable structure exists; useful predictions refine it.

The same numerical pattern can require different treatment at different stages.
Mathematically, the DiT is learning a **velocity field** `v(x,t,y)`: a rule whose
answer depends on state, time, and condition.

Do not confuse:

```text
t   = information given to the DiT about the current stage
dt  = step size used afterward by the Euler sampler
```

## 3. Why normalize activations?

A token contains 128 working activation values. As many layers multiply and add
values, their overall magnitude can drift. Very large or tiny scales make
training harder and are especially awkward for fixed-range int8 arithmetic.

Pico-Faces uses **RMSNorm**. It measures the root-mean-square magnitude of a
token and divides by it. It controls scale without subtracting the mean.

For example, `[3, 4]` has RMS:

```text
sqrt((3^2 + 4^2) / 2) = sqrt(12.5) = about 3.54
```

After RMS normalization:

```text
[3/3.54, 4/3.54] = about [0.85, 1.13]
```

The proportional pattern remains, but its overall magnitude is predictable.
This helps keep successive blocks numerically well-behaved; it does not mean
that every token becomes identical.

“Prevents a random walk” is a useful first intuition, but not quite the formal
job. RMSNorm directly controls the scale presented to the next transformation.
That stabilizes activations and training even though residual additions can
still change the unnormalized residual stream.

## 4. What AdaLN changes

The model converts time and class into one condition vector:

```text
condition = time representation + class representation
```

For each block, a small conditioning network turns that vector into six sets of
controls:

```text
attention scale, attention shift, attention gate
MLP scale,       MLP shift,       MLP gate
```

### What are the two branches?

A transformer block has two main residual work paths:

1. The **attention branch** mixes information among the 64 patch tokens. It
   lets one region use information from other regions.
2. The **MLP branch** transforms the 128 features inside each token separately.
   It does not itself mix one patch token with another.

They are called branches because each computes a proposed change along a side
path while the original `x` also travels directly to the final addition:

```text
                 branch proposes change
                /                       \
input x --------                         + -------- output
        \-------------------------------/
                 unchanged residual path
```

For one branch, the idea is:

```text
normalized = RMSNorm(x)
adjusted   = normalized * (1 + scale) + shift
proposal   = attention_or_MLP(adjusted)
output     = x + gate * proposal
```

Thus the condition has three kinds of control:

- **Scale:** strengthen or weaken normalized features.
- **Shift:** move features up or down.
- **Gate:** control how much of the branch's proposed change is added back.

A gate is simply a learned multiplier. In this model it is a vector, so
different features can be opened by different amounts:

```text
gate = 0     -> add none of the proposal
gate = 0.25  -> add one quarter of the proposal
gate = 1     -> add the full proposal
gate < 0     -> add it in the opposite direction
```

Time and class do not directly “shift the final velocity.” They alter processing
inside every block, and all those conditioned transformations eventually
produce the velocity prediction.

### AdaLN controls are not CFG values

AdaLN's scale, shift, and gate are not synonyms for CFG's null prediction,
conditional prediction, and guidance weight.

- **AdaLN is inside one DiT evaluation.** It uses that evaluation's time and
  class to control attention and MLP branches throughout the model.
- **CFG is outside the DiT.** It evaluates the complete AdaLN-conditioned DiT
  once with the null class and once with the requested class, then combines the
  two final velocity tensors using `w`.

Therefore AdaLN is not layered on top of CFG. The nesting is the other way
around: CFG wraps two calls to a DiT that internally uses AdaLN.

## 5. What “Zero” means

At the beginning of training, the small network that produces scale, shift, and
gate controls is initialized to output zeros:

```text
scale = 0
shift = 0
gate  = 0
```

Substituting those values:

```text
adjusted = normalized * (1 + 0) + 0 = normalized
output   = x + 0 * proposal = x
```

So a new block initially behaves like a pass-through: its residual branch makes
no change. Training gradually moves the gates, scales, and shifts away from zero
when doing so reduces error. This gives a deep stack a stable starting point.

The “Zero” therefore refers to zero-initialized modulation and residual gates,
not zero activations and not a null class.

## 6. Why lookup tables work on the Pico

The full training model computes controls from time and class with small neural
networks. The deployed sampler uses only eight baked timesteps and five class
values across a fixed number of blocks. Export can calculate the required
controls ahead of time and store them in tables.

At runtime the Pico selects a row by timestep, class, and block instead of
running the conditioning networks. This uses flash to save computation. It is
not primarily a space saving—the tables occupy substantial flash—but it removes
runtime conditioning work and makes integer inference simpler.

## Run the zero-gate demonstration

```powershell
python .\lesson_08a_adaln.py
```

## Checkpoint

1. In this project, is a class something the DiT predicts, or a control given
   to it?
2. What different information do `t` and `dt` represent?
3. If the AdaLN residual gate is zero, what happens to the block's proposed
   change?

