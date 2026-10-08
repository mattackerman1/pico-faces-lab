# Lesson 12 — How the model fits and runs on the Pico 2

## Goal

Trace one generation through flash, SRAM, both Cortex-M33 cores, and USB, and
understand how phase-exclusive memory reuse makes a 2.57 MB model practical on
a microcontroller with 520 KB of SRAM.

## The hardware resources

The relevant resources are:

```text
4 MB external QSPI flash   firmware instructions + embedded model blob
520 KB on-chip SRAM        current activations, staged weights, scratch, image
two Cortex-M33 cores       parallel integer computation
USB-CDC                    commands in, raw RGB image out
```

Flash is large enough for the complete fast model but slower than SRAM. SRAM is
fast but far too small to hold the 2.57 MB blob plus all intermediate values.
The firmware therefore leaves the model in flash and copies only the weight
section required by the current operation into reusable SRAM buffers.

## 1. Build time: embed the blob in flash

`firmware/model_blob.S` uses the assembler's `.incbin` directive to insert the
selected `model.bin` directly into the firmware's read-only flash image:

```text
compiled firmware code
+ model.bin bytes
        |
        v
      UF2
        |
        v
Pico external flash
```

The symbols `rf_model_blob` and `rf_model_blob_end` mark its boundaries. At
boot, `rf_model_load` validates the header and creates pointers into the blob;
it does not copy the entire model into SRAM.

## 2. Execute-in-place flash

The RP2350 can address external flash as if it were memory. This is called
**execute in place**, or XIP. Firmware instructions and model bytes can be read
through the QMI flash interface, assisted by a 16 KB XIP cache.

Large transformer matrices are much bigger than that cache. Reading one matrix
from flash separately for every token would repeatedly evict and refetch it.
Instead, the engine copies each needed matrix into an SRAM staging slot once,
then reuses that fast SRAM copy for all 64 tokens.

The checked-in build defaults to a synchronous `memcpy` staging path:

```text
copy current weights from flash to SRAM
wait for copy to finish
compute with those weights
```

An optional build flag, `RF_STAGE_DMA=1`, replaces this with paced DMA so a
future matrix can be copied while computation continues. The DMA path limits
flash traffic so it does not starve instruction fetches. Both paths copy the
same bytes and produce byte-identical numerical results; they differ only in
when and how the data moves.

## 3. The 256 KB reusable arena

The engine reserves two 128 KB SRAM halves:

```text
rf_arena[0] = 128 KB
rf_arena[1] = 128 KB
total       = 256 KB
```

During a DiT step, those halves contain fixed weight-staging slots:

```text
arena 0: FC1 64 KB | FC2 64 KB
arena 1: QKV 48 KB | projection 16 KB | embedding/final slots
```

After all DiT steps finish, those transformer weights are no longer needed.
The VAE decoder reuses the same two arena halves as ping-pong activation
buffers:

```text
decoder layer N reads arena 0 and writes arena 1
decoder layer N+1 reads arena 1 and writes arena 0
```

Only the current and next decoder feature maps need to exist at once.

This is a central embedded technique: memory whose lifetimes do not overlap can
occupy the same physical bytes.

## 4. More phase-exclusive reuse

The DiT's query/key/value and attention buffers are needed during transformer
steps but idle during VAE decoding. The decoder overlays its sparse-row
compaction workspace on that same memory.

The optional VGA framebuffer also borrows arena storage while inference is
idle, then invalidates it before the engine reuses the arena. We are not using
VGA, but the architecture illustrates the same lifetime-based reuse.

Compile-time static assertions check that a future decoder's scratch structures
still fit in the memory they borrow. A model change that exceeds the reserved
space therefore becomes a build error rather than silent memory corruption.

## 5. Both cores split independent ranges

`rf_par_for` divides a token or row range in half:

```text
core 0: [0, n/2)
core 1: [n/2, n)
```

Core 0 sends a “go” message through the inter-core FIFO, performs its own half,
then waits for core 1's “done” message. Each core writes to a disjoint output
range, so no locks are required inside the work.

Parallel execution is safe only where one output does not depend on another
output being updated first. The algorithm's step and layer ordering remains
sequential:

```text
within one suitable operation: two cores work in parallel
between dependent layers/steps: wait, then advance together
```

## 6. One complete generation

```text
USB command arrives
        |
        v
model pointers refer to blob in XIP flash
        |
        v
PRNG creates the initial latent in SRAM
        |
        v
for each Euler step and DiT block:
  copy required matrix flash -> arena slot
  split token work across both cores
  retain only the required current activations
        |
        v
reuse arena as VAE ping-pong buffers
        |
        v
write 49,152 RGB bytes to image buffer
        |
        v
send header + RGB + CRC + timing over USB
```

## Checkpoint

Explain these in your own words:

1. Why is `model.bin` kept in flash instead of copied completely into SRAM?
2. Why copy a weight matrix into SRAM once per operation instead of reading it
   directly from flash for every token?
3. How can the same 256 KB arena hold transformer weights during one phase and
   VAE activations during another without destroying needed data?
4. Why can two cores process separate token or image-row ranges without locks,
   while transformer layers and Euler steps still have to remain ordered?
5. What does optional DMA change, and what does it deliberately not change?
