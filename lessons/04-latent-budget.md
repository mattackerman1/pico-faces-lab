# Lesson 04 — The three-part image generator

This lesson introduces the vocabulary before doing any calculations. Nothing
here is a test; the checkpoint is something we complete together.

## Outcome

Understand the jobs of the VAE and DiT, then see why the latent contains 24
times fewer values than the final image.

## The whole system first

The model uses a small compressed representation called a **latent** as its
internal image format:

```text
TRAINING ON A COMPUTER

real face               compressed face
128 x 128 x 3  ──VAE encoder──>  16 x 16 x 8
                                      |
                                      | examples used to train the DiT
                                      v


GENERATION ON THE MICROCONTROLLER

random latent           refined latent             visible image
16 x 16 x 8  ──DiT──>   16 x 16 x 8  ──VAE decoder──> 128 x 128 x 3
                 repeated steps
```

The microcontroller needs the DiT and VAE decoder. It does not need the VAE
encoder because it is generating images, not compressing existing ones.

## What is a VAE?

**VAE** means **Variational Autoencoder**. It is not “variable autoencoder.”

An autoencoder has two halves:

- The **encoder** compresses a real image into a smaller collection of numbers,
  the latent.
- The **decoder** reconstructs visible pixels from that latent.

During training, the two halves learn a useful compressed “language” for faces.
The word *variational* refers to training the latent space so it is smooth and
organized enough to sample and generate from. We do not need its detailed math
yet.

In this project, the large encoder is a training tool on the PC. Only the small
decoder is placed on the microcontroller.

## What is a DiT?

**DiT** means **Diffusion Transformer**.

Its input is a noisy latent, not a visible RGB image. At each generation step,
it predicts how that latent should change to become more like a plausible face.
The program applies a small part of that predicted change and calls the DiT
again. The released model can use 1, 2, 4, or 8 such steps.

So the responsibilities are different:

- **DiT:** create/refine the compressed description of a face.
- **VAE decoder:** translate the finished compressed description into RGB
  pixels.

## Why is the compression factor 24?

The RGB image contains:

```text
128 x 128 x 3 = 49,152 values
```

The latent contains:

```text
16 x 16 x 8 = 2,048 values
```

Therefore:

```text
49,152 / 2,048 = 24
```

The easy-to-miss detail is the channel count:

- Width shrinks by 8: `128 / 16 = 8`.
- Height shrinks by 8: `128 / 16 = 8`.
- That makes 64 times fewer spatial positions: `8 x 8 = 64`.
- But each latent position has 8 channels instead of 3.
- Net reduction: `64 x (3 / 8) = 24`.

So 64 describes the reduction in **spatial positions**, while 24 describes the
reduction in the **total number of stored values**.

The number 24 was not forced by a law of transformers. The author chose the
latent dimensions as an engineering tradeoff: smaller is cheaper to generate,
but compressing too aggressively loses facial detail. The article reports that
a larger decoder helped only marginally, evidence that the compressed latent
itself had become a bottleneck.

## What is a channel?

A channel is one number stored at every spatial location.

For an RGB image, each pixel contains three numbers with meanings chosen by
humans: red, green, and blue. The image therefore has three channels.

For the latent, each of its 16 x 16 locations contains eight numbers. These are
also channels in the structural sense, but they are **learned latent channels**,
not eight visible colors. No channel necessarily means something simple such
as “eyes,” “brightness,” or “smile.” The encoder learns a distributed numerical
code, and the decoder learns how to interpret all eight values together.

One useful mental picture is:

```text
RGB pixel:       [red, green, blue]
Latent position: [z0, z1, z2, z3, z4, z5, z6, z7]
```

The symbols `z0` through `z7` mean learned latent values whose individual
meanings are not named for us.

## How do patches become tokens?

The latent is 16 positions wide and 16 positions tall, with 8 channels at every
position. The DiT groups it into 2 x 2 spatial patches:

```text
16 positions / 2 positions per patch = 8 patches across
16 positions / 2 positions per patch = 8 patches down

8 x 8 = 64 patches total
```

There are therefore **8 patches per row**, not 8 patches total. One patch takes
four neighboring latent positions and includes all eight channels at each one:

```text
top-left:      8 values    top-right:     8 values
bottom-left:   8 values    bottom-right:  8 values

4 positions x 8 channels = 32 values in one patch
```

Those 32 values are flattened into one list. A learned embedding transforms
that list into one 128-number transformer token. The DiT therefore operates on
64 tokens.

## Why use 2 x 2 patches?

Patch size is another engineering choice, not a mathematical requirement:

- With 1 x 1 patches, the 16 x 16 latent would produce 256 tokens. That retains
  fine spatial granularity, but its attention matrix has `256 x 256 = 65,536`
  entries per head.
- With 2 x 2 patches, it produces 64 tokens and an attention matrix with
  `64 x 64 = 4,096` entries—16 times fewer.
- With 4 x 4 patches, it would produce only 16 tokens, but each token would
  cover a much larger area and make fine spatial relationships harder for the
  small model to express.

Thus 2 x 2 is a balance: much cheaper attention than 1 x 1, while retaining
more spatial detail than a larger patch. The repository establishes that this
is the chosen architecture; this tradeoff explanation is our engineering
interpretation, not a claim that 2 x 2 is the only valid choice.

## Verify the arithmetic

Run this in PowerShell:

```powershell
python -c "pixels=128*128*3; latent=16*16*8; across=16//2; print({'pixels': pixels, 'latent': latent, 'compression': pixels/latent, 'patches_across': across, 'tokens': across*across})"
```

Expected values:

```text
pixels: 49152
latent: 2048
compression: 24.0
patches_across: 8
tokens: 64
```

## Guided checkpoint

Here are the answers we were aiming for:

1. The compression is 24 because width and height each shrink eightfold while
   channels increase from 3 to 8: `8 x 8 x 3 / 8 = 24`.
2. A 16 x 16 latent divided into 2 x 2 patches gives 8 patches across and 8
   down, or 64 tokens.
3. The DiT builds a face in the compressed latent representation. The VAE
   decoder turns that finished representation into 128 x 128 RGB pixels.

Before moving on, try explaining the pipeline in your own words using only
these terms: **noise, DiT, latent, VAE decoder, pixels**. Two sentences are
enough.

