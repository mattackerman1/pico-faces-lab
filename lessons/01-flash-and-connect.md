# Lesson 01 — Flash and connect

Do this lesson only after we confirm the board model in Lesson 00.

## Outcome

Flash one released firmware image and identify the board's USB serial port.

## Why we start with released firmware

It separates hardware and USB setup from compiler and model-export problems.
Once the known-good image runs, later failures are much easier to localize.

## Steps

1. Open the upstream [`uf2/`](https://github.com/cpldcpu/pico-faces/tree/main/uf2)
   folder.
2. Start with the fast `m3_long_cfg` UF2 unless we decide otherwise. It is the
   quicker feedback loop; the upstream project reports about 4.3 seconds for
   four guided steps.
3. Disconnect the RP2350 board.
4. Hold its BOOTSEL button while reconnecting USB. Release BOOTSEL after the
   mass-storage drive appears. The drive is normally named `RPI-RP2`.
5. Copy the selected `.uf2` file to that drive. The board should reboot and the
   drive should disappear automatically.
6. Find the new serial port in PowerShell:

   ```powershell
   [System.IO.Ports.SerialPort]::GetPortNames()
   ```

   If several ports appear, run the command once with the board disconnected
   and once with it connected; the new entry is the board.

## Verification

Record these in [`../PROGRESS.md`](../PROGRESS.md):

- UF2 filename
- Board model
- Serial port, such as `COM10`
- Whether the board rebooted after the copy
- Any unexpected LED, drive, or USB behavior

## Checkpoint

Success means a new serial port appears after flashing. Stop before installing
the viewer dependencies; that is Lesson 02.

## Recovery notes

- If `RPI-RP2` does not appear, suspect the USB cable first.
- If the UF2 copy fails, confirm the firmware targets RP2350/Pico 2 rather than
  the older RP2040/Pico.
- If no serial port appears after reboot, capture the board name, UF2 filename,
  and Windows Device Manager status before trying another image.

