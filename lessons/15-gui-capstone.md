# Lesson 15 — Implement and validate the GUI capstone

## What we built

`pico_faces_gui.py` provides a virtual 128 x 128 LCD and physical-style
controls for class, seed, Euler steps, and guidance. Serial work happens on a
background thread so the interface remains responsive while the Pico runs.

The division of responsibility remains:

- **PC:** collect button choices, send a command, receive bytes, verify CRC,
  enlarge the image, and display it.
- **Pico 2:** create noise, run DiT sampling, decode the latent, and produce the
  RGB pixels.

## Start it

From PowerShell in the project directory:

```powershell
.\launch_gui.ps1
```

The default controls reproduce our baseline: COM3, seed 42, class 1, four
steps, and CFG 4.

## Validation completed

- Python compilation: passed.
- Protocol helper self-test: passed.
- Real COM3 request through the GUI's serial client: passed.
- Returned frame: 128 x 128 x 3.
- Device time: 5,429 ms.
- Device and host CRC: `82e03d24` (match).
- GUI startup: passed; the interactive window remained running.

## Final visual check

Click **Generate** in the open window. The face should replace the placeholder,
the status should say the CRC was verified, and the details should report seed
42, the selected class, generation time, and CRC `82e03d24`.
