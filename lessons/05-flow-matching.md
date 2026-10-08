# Lesson 05 — Flow matching: learning a direction

## Outcome

Understand what the DiT is trained to predict and how repeated predictions move
random noise toward a plausible latent.

We begin with one number. The real model performs the same kind of arithmetic
on all 2,048 latent values at once.

## Step 1: Define a start and destination

Suppose one training example pairs:

```text
noise: x0 = -3
clean example: x1 = 5
```

During training, both are known. Their straight-line velocity is:

```text
v = x1 - x0
v = 5 - (-3)
v = 8
```

Velocity here means the total direction and distance from the noise value to
the clean value. It is not physical motion or pixels per second.

## Step 2: Pick a point along the path

Time `t` runs from 0 to 1. The training code mixes the endpoints:

```text
xt = t*x1 + (1-t)*x0
```

For `t = 0.25`:

```text
xt = 0.25*5 + 0.75*(-3)
xt = -1
```

At this training instant, the model receives approximately:

```text
current state xt = -1
time t = 0.25
optional class condition
```

Its desired answer—the label used to calculate training error—is `v = 8`.

## Step 3: What the DiT learns

The training program repeats this with many random noise samples, clean face
latents, times, and class conditions. The DiT adjusts its weights so its
predicted velocity approaches the known target velocity.

It is helpful to think of the trained DiT as answering:

> Given where I am, how far through the journey I am, and the requested class,
> which direction should this latent move now?

The actual answer is a 16 x 16 x 8 velocity tensor—one predicted change for
each of the 2,048 latent values.

## Step 4: Training versus generation

During training, the clean destination `x1` is known, so the correct velocity
can be calculated.

During generation, `x1` is unknown—that is the image we are trying to create.
We begin with noise and ask the trained DiT to predict a velocity. The sampler
takes a small step, advances time, and asks again:

```text
x_next = x_current + step_size * predicted_velocity
```

This is an Euler integration step. Multiple steps let the prediction adapt as
the latent changes. In the Pico model, the available schedules use 1, 2, 4, or
8 steps.

The model does **not** save and replay a velocity seen during training. Training
adjusts its weights so it becomes a velocity-predicting function:

```text
predicted_velocity = DiT(current_latent, current_time, condition)
```

The sampler calls that function again at every step.

## What exactly is an Euler step?

Euler's method is a small-update rule for following a direction that may change:

```text
new state = current state + time step x current velocity
```

For four steps between time 0 and 1, the time step is:

```text
dt = 1 / 4 = 0.25
```

In our deliberately simple example, the perfect velocity is always 8, so each
update adds `0.25 x 8 = 2`:

| Step | Current time | State before | Predicted velocity | Change | State after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | -3 | 8 | +2 | -1 |
| 2 | 0.25 | -1 | 8 | +2 | 1 |
| 3 | 0.50 | 1 | 8 | +2 | 3 |
| 4 | 0.75 | 3 | 8 | +2 | 5 |

At each row, imagine that the DiT is called again using the new state and time.

Our straight-line toy is unusually easy: because its velocity is perfectly
known and constant, one full step would also land exactly on 5. A real trained
DiT is estimating a complicated velocity field across many possible faces. Its
prediction can change as facial structure emerges and time advances. Smaller,
repeated Euler steps let the sampler ask for a corrected direction along the
way. This is why the toy explains the update arithmetic but does not by itself
demonstrate the quality benefit of more steps.

## Step 5: Run the tiny demonstration

From the project folder:

```powershell
python .\lesson_05_flow_matching.py
```

The script needs only Python's standard library. It prints the exact straight
path using four Euler steps and then verifies that the endpoint is 5.

This toy asks a stand-in “perfect model” for the velocity on every step. That
stand-in returns the same value each time. A trained model instead supplies a
new estimate, which is why additional sampling steps can reveal and correct
more detail.

## Checkpoint

Answer in your own words:

1. During training, how can we calculate the correct velocity?
2. During generation, why can we not calculate that velocity directly?
3. For `x0 = -3`, `x1 = 5`, and `t = 0.25`, what are `v` and `xt`?

