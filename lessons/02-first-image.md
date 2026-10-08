# Lesson 02 — Generate the first image

Do this lesson only after Lesson 01 has produced a serial port.

## Outcome

Request an image over USB, save it as PNG, and record generation time.

## Planned steps

We will clone the upstream repository into `upstream/pico-faces`, create an
isolated Python environment, install the viewer dependencies, and run a command
of this form with the actual serial port:

```powershell
python viewer/view_serial.py --port COM10 --seed 42 --steps 4 --class 1 --cfg 4 --show
```

The parameters mean:

- `seed 42`: fixes the starting noise, making the result reproducible.
- `steps 4`: uses four Euler integration steps.
- `class 1`: requests the upstream female/smiling condition.
- `cfg 4`: applies the balanced classifier-free guidance setting.
- `show`: opens a nearest-neighbor-upscaled view after saving the real image.

We will enter the exact commands together so environment or USB issues can be
handled one at a time.

## Verification

Success means all of the following are true:

- The viewer reports a completed frame and timing.
- A 128 x 128 PNG exists in the output directory.
- Repeating the same request produces the same device-reported CRC.

## Checkpoint

Save the image, timing, and CRC in [`../PROGRESS.md`](../PROGRESS.md). That gives
us a baseline for every later experiment.

