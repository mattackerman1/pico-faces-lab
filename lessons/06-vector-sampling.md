# Lesson 06 — From one number to a latent

## Outcome

Follow Euler sampling on a tiny 2 x 2 latent and connect it to the Pico model's
16 x 16 x 8 latent.

## Step 1: Replace the scalar with a tiny latent

Our previous state was one number. Now imagine a latent with four positions:

```text
current latent x = [ -2.00,  1.00 ]
                   [  0.00,  3.00 ]
```

This is intentionally tiny. The Pico model has eight values at every one of its
16 x 16 positions, for 2,048 values total.

## Step 2: The DiT returns the same shape

Given the current latent, time, and class, our toy model predicts:

```text
predicted velocity = [  6.00, -3.00 ]
                     [  1.50, -4.50 ]
```

The velocity has the same shape as the latent. Every value has a proposed
direction and magnitude. In the real model, a prediction at one position can
depend on information from other positions because the transformer mixes
information among tokens.

## Step 3: Apply one Euler step

With four total steps, `dt = 0.25`. Multiply every velocity value by `dt`:

```text
change = 0.25 x velocity

change = [  1.50, -0.75 ]
         [  0.375, -1.125 ]
```

Add the change element by element:

```text
next latent = current latent + change

next latent = [ -0.50,  0.25  ]
              [  0.375, 1.875 ]
```

No value changes shape or location. The update is elementwise, even though the
DiT used the whole latent to decide what velocities to predict.

## Step 4: Repeat the loop

After the update:

1. Advance time from `0.00` to `0.25`.
2. Give the new latent and time to the DiT.
3. Receive a new velocity matrix.
4. Multiply it by `dt` and add it to the latent.

Repeat until time reaches 1. The VAE decoder receives only the final latent.
Intermediate latents do not need to look like visible images.

## Step 5: Run the demonstration

```powershell
python .\lesson_06_vector_sampling.py
```

The script uses a transparent toy velocity predictor that pulls the four
numbers toward a fixed example. It is not a neural network and is not intended
to imitate the DiT's internal calculations. Its purpose is to expose the
sampler loop and tensor-shaped arithmetic.

Look for this sequence on every step:

```text
current latent -> predicted velocity -> scaled change -> next latent
```

## Scaling up to Pico-Faces

The same update rule applies to the real latent:

```text
tiny lesson:  x and velocity each have shape 2 x 2
Pico-Faces:   x and velocity each have shape 16 x 16 x 8
```

The Euler sampler itself is simple. Nearly all learned intelligence is in the
DiT that predicts the velocity tensor.

## Checkpoint

1. Why must the DiT's output have the same shape as the current latent?
2. For current value `1.00`, velocity `-3.00`, and `dt = 0.25`, what is the next
   value?
3. Does the VAE decoder process every intermediate latent or only the final
   one?

