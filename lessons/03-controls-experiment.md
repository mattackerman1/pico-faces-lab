# Lesson 03 — Explore the generation controls

## Outcome

Change one variable at a time and explain the speed/quality tradeoffs from
observed results rather than intuition alone.

## Experiment A: integration steps

Hold seed and class fixed. Generate with 1, 2, 4, and 8 steps. Record time and
compare when recognizable structure and fine details appear.

## Experiment B: conditioning

Hold seed, steps, and guidance fixed. Generate classes 0 through 4. The released
labels are:

| Class | Condition |
| --- | --- |
| 0 | female, neutral |
| 1 | female, smiling |
| 2 | male, neutral |
| 3 | male, smiling |
| 4 | unconditional/null |

These are statistical conditioning labels, not hard guarantees about an
individual output.

## Experiment C: classifier-free guidance

Hold seed, class, and steps fixed. Compare plain sampling with guidance weights
4, 6, and 8. Guided sampling evaluates the DiT twice per step, so expect roughly
twice the generation work.

## Results table

Copy this into [`../PROGRESS.md`](../PROGRESS.md) and add rows as needed:

| Seed | Steps | Class | CFG | Time | CRC | Observation |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 42 | 4 | 1 | 4 | | | baseline |

## Checkpoint

Write three sentences:

1. What extra steps changed.
2. What stronger CFG changed.
3. Which setting gives the best speed/quality compromise on your board.

