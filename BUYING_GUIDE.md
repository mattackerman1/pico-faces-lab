# Minimal buying guide

## Recommended board

Buy an **official Raspberry Pi Pico 2 H** (RP2350, with headers).

Why this version:

- It has the RP2350, 520 KB SRAM, and 4 MB flash required by the reference
  project.
- The `H` version is electrically the same basic Pico 2 but has headers already
  installed, which makes later physical extensions easier without requiring
  soldering on day one.
- The non-wireless board keeps our first experiment close to the upstream
  firmware target. Wi-Fi and Bluetooth are not needed to generate or transfer
  images over USB.

Be careful not to buy the original **Pico**, **Pico H**, **Pico W**, or **Pico
WH**. Those are RP2040 boards, not the RP2350-based Pico 2 family.

## Cable

Buy a **USB data cable with a Micro-B plug for the Pico 2 end** and the plug your
computer needs on the other end (usually USB-A or USB-C).

Some inexpensive cables supply power only. The listing should explicitly say
that the cable supports data or synchronization.

## Do not buy yet

- VGA hardware—the reference firmware can return completed images over USB.
- A debugger—the BOOTSEL/UF2 workflow is enough initially.
- A display—we should first reproduce the known-good USB result, then choose a
  display whose interface, resolution, and framebuffer needs fit our capstone.
- A breadboard or jumper wires—they are useful later but unnecessary for the
  first milestone.

## What “self-contained” will mean

The first demo is computationally self-contained: generation happens entirely
on the RP2350, while a PC requests the image and displays the received pixels.
A later capstone can remove that host dependency by adding local controls and a
small display. We will make that hardware decision only after measuring the
reference firmware.

## Official references

- [Raspberry Pi Pico 2 product page](https://www.raspberrypi.com/products/raspberry-pi-pico-2/)
- [Raspberry Pi Pico-series documentation](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html)

