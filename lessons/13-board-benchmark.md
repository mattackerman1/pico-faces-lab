# Lesson 13 — Benchmark the Pico 2

## Result

We measured plain sampling and CFG 4 at 1, 2, 4, and 8 Euler steps. Each
configuration ran three times with seed 42 and class 1.

| Mode | Steps | Mean time | Observed range | CRC32 |
| --- | ---: | ---: | ---: | --- |
| Plain | 1 | 2,172.3 ms | 2,170–2,175 ms | `f2bb1ca0` |
| Plain | 2 | 2,553.0 ms | 2,549–2,555 ms | `a9451b02` |
| Plain | 4 | 3,469.0 ms | 3,467–3,472 ms | `bde6bba1` |
| Plain | 8 | 5,358.7 ms | 5,356–5,360 ms | `e93799d5` |
| CFG 4 | 1 | 2,582.0 ms | 2,581–2,583 ms | `fbbd9970` |
| CFG 4 | 2 | 3,533.0 ms | 3,531–3,534 ms | `31c70032` |
| CFG 4 | 4 | 5,436.7 ms | 5,434–5,441 ms | `82e03d24` |
| CFG 4 | 8 | 9,210.7 ms | 9,208–9,215 ms | `ddc2cf3d` |

Every repetition of a configuration produced the same CRC, so the generated
bytes were deterministic. The small timing ranges show that the measurements
were also stable.

## Timing model

A straight line fitted to the four step counts gives:

```text
plain time ≈ 1,666.5 ms + 459.1 ms × steps
CFG 4 time ≈ 1,640.0 ms + 946.8 ms × steps
```

The intercept is an estimate of fixed generation work: initialization, latent
setup, and VAE decoding, plus other work that does not grow with the Euler-step
count. It does not separately measure those parts.

The slope estimates the incremental cost of one Euler step. Plain sampling
runs the DiT once and costs about 0.46 seconds per step. CFG runs it twice and
costs about 0.95 seconds per step—almost exactly twice as much.

## Takeaway

The one-step plain run is the best single quick test, but several plain step
counts are what let us separate fixed cost from per-step cost. Our preferred
quality baseline remains four steps with CFG 4: about 5.44 seconds.

Raw measurements are in `device-output/lesson13-benchmark/results.csv`. The
repeatable runner is `lesson_13_benchmark.ps1`.

## Checkpoint

Why can a one-step run not, by itself, tell us how much time belongs to fixed
work and how much belongs to the Euler step?
