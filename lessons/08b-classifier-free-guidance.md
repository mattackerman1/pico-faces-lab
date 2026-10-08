# Lesson 08B — Classifier-free guidance

Complete Lesson 08A first.

## Outcome

Understand what guidance is for, why it is called classifier-free, and why it
requires two DiT evaluations per Euler step.

## The problem guidance addresses

A normal conditional DiT prediction responds to the requested class, but the
effect can be weak because the model must also produce plausible and varied
faces. Classifier-free guidance amplifies the part of the predicted direction
that is specifically associated with the requested class.

## Two predictions, one model

Use the same DiT weights, current latent, and time twice:

```text
v_null = DiT(x, t, null class)
v_cond = DiT(x, t, requested class)
```

Their difference estimates the class-specific direction:

```text
class direction = v_cond - v_null
```

The sampler constructs a guided velocity:

```text
v_guided = v_null + w * class direction
```

The DiT has already made both predictions. CFG is the arithmetic that combines
them; it is not a third velocity-prediction method.

## Why “classifier-free” if it uses classes?

Earlier guidance methods used a separate image classifier network to inspect a
noisy sample and provide a direction toward a desired label. That required
training and running an additional classifier.

Classifier-free guidance needs no separate classifier. The generator itself is
trained to make both conditional and unconditional predictions. “Classifier-
free” means **free of an external classifier**, not free of class labels.

## What the guidance weight does

If:

```text
v_null = 2
v_cond = 3
```

then the class direction is `1`. With `w = 4`:

```text
v_guided = 2 + 4*(3 - 2) = 6
```

The weight multiplies only the conditional-minus-null difference. Larger values
usually strengthen class adherence and structure, but can reduce variety or
produce exaggerated artifacts. This is guidance at sampling time, not model
overfitting; the trained weights are not being changed.

## What doubles

CFG doubles **DiT evaluations**, not Euler steps:

```text
8 Euler steps without CFG =  8 DiT evaluations
8 Euler steps with CFG    = 16 DiT evaluations
```

The two evaluations are logically separate. On a memory-constrained Pico they
do not imply two copies of the model or necessarily simultaneous parallel
execution; the same model can evaluate the null and conditional cases in turn.

The VAE decoder still runs once, after the final guided latent.

## Run the CFG calculator

```powershell
python .\lesson_08_cfg.py
```

## Checkpoint

1. What does “classifier-free” mean?
2. Does CFG train another model or combine two outputs from the same DiT?
3. Why can a high guidance weight reduce variety?

