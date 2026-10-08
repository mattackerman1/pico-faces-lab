# Lesson 00 — Orient the project

## Outcome

Choose a sensible starting track and confirm what equipment is available. This
lesson should take about 10 minutes and changes nothing on the computer or
board.

## The project in five facts

1. The device generates a compressed 16 x 16 x 8 latent, not 128 x 128 pixels
   directly.
2. A diffusion transformer predicts a velocity that moves noise toward a clean
   latent over 1, 2, 4, or 8 Euler steps.
3. A small VAE decoder turns the final latent into a 128 x 128 RGB image.
4. Post-training int8 quantization, flash streaming, DMA, and two CPU cores make
   the model fit and run on the RP2350.
5. USB output is enough for the project. VGA is an optional display path.

## Choose a track

Pick the first track whose requirements match what is available:

### Track A — Board-first (recommended)

Use this if you have an RP2350 board and a USB data cable. We begin by flashing
the released UF2, generating images, and measuring the model before studying
its internals.

### Track B — Desktop-first

Use this if the board has not arrived. We begin with the model concepts and the
portable desktop inference engine, then switch to hardware later.

### Track C — Full training

Treat this as an optional extension, not the default. The released training
pipeline uses FFHQ-derived data and substantial CUDA compute. We will still
learn every important model idea without retraining the flagship model.

## Equipment inventory

Answer these in [`../PROGRESS.md`](../PROGRESS.md):

- Which exact RP2350 board do you have, if any?
- Do you have a known data-capable USB cable?
- What operating system and Python version are you using?
- Do you have the optional Pimoroni VGA Demo Base?
- Is a CUDA GPU available? If so, which one and how much VRAM?
- Which matters more: learning the ML, learning embedded optimization, or
  getting a working demo quickly?

To check Python in PowerShell:

```powershell
python --version
```

If that is not recognized, try:

```powershell
py --version
```

## Checkpoint

Stop here and share the inventory answers. Together we will select Track A or
B and start Lesson 01 with instructions tailored to the exact board and host.

## Optional reading

- [Project article](https://cpldcpu.github.io/2026/08/28/ai-image-generation-on-a-rp-pico-2-microcontroller/)
- [Upstream repository](https://github.com/cpldcpu/pico-faces)

