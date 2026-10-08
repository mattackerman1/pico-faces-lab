# Lesson 14 — Capstone design

## Selected capstone

Build a desktop GUI that simulates a future self-contained LCD-and-button
interface while the real Pico 2 performs all image generation.

```text
virtual buttons in GUI
        |
        | generation command over USB serial
        v
Pico 2: PRNG -> DiT sampling -> VAE decoding
        |
        | raw RGB image over USB serial
        v
virtual 128 x 128 LCD in GUI
```

The GUI is not an AI simulator: it is a simulated control panel and display.
The model and inference engine continue to run entirely on the RP2350.

## First version

- A 128 x 128 virtual LCD, enlarged with nearest-neighbor scaling so individual
  pixels remain crisp.
- Four class-selection buttons plus an unconditional option.
- Controls for seed, step count, and plain/CFG mode.
- A Generate button.
- Status showing generation time, CRC, serial state, and errors.
- Sensible defaults: class 1, seed 42, four steps, CFG 4.

The controls should resemble actions that could later become physical buttons.
Serial communication will be kept separate from the GUI code so a future LCD
and GPIO interface can reuse the same behavior rather than redefining it.

## Success criteria

1. The application connects to the Pico on `COM3`.
2. Clicking Generate sends the selected settings to the Pico.
3. The Pico—not the PC—generates the image.
4. The returned RGB frame appears on the virtual LCD.
5. Timing, CRC, busy state, and communication failures are visible.
6. Repeating identical settings produces the same CRC and image.

## Why this capstone

It tests the interaction design without requiring an LCD, buttons, wiring, or
firmware display driver. It is not yet physically self-contained, but it gives
us an executable specification for that later device.
