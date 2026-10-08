"""Expose Euler sampling with a tiny 2x2 vector instead of an image latent."""

from typing import TypeAlias

Vector: TypeAlias = list[float]

NOISE: Vector = [-2.0, 1.0, 0.0, 3.0]
TOY_ATTRACTOR: Vector = [2.0, -1.0, 1.0, 0.0]
STEPS = 4


def toy_dit(current: Vector, time: float) -> Vector:
    """Transparent stand-in for a learned, time-conditioned DiT."""
    strength = 1.5 - 0.5 * time
    return [strength * (target - value) for value, target in zip(current, TOY_ATTRACTOR)]


def scaled(values: Vector, amount: float) -> Vector:
    return [amount * value for value in values]


def added(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right)]


def grid(values: Vector) -> str:
    return f"[{values[0]:7.3f}, {values[1]:7.3f}]\n[{values[2]:7.3f}, {values[3]:7.3f}]"


def main() -> None:
    dt = 1.0 / STEPS
    current = NOISE.copy()

    print("Tiny 2x2 latent Euler sampler")
    print(f"steps={STEPS}, dt={dt}\n")

    for step in range(STEPS):
        time = step * dt
        velocity = toy_dit(current, time)
        change = scaled(velocity, dt)
        next_state = added(current, change)

        print(f"STEP {step + 1}: t={time:.2f} -> {time + dt:.2f}")
        print("current latent:")
        print(grid(current))
        print("predicted velocity:")
        print(grid(velocity))
        print("scaled change (dt * velocity):")
        print(grid(change))
        print("next latent:")
        print(grid(next_state))
        print()

        current = next_state

    print("Final latent (this is what the VAE decoder would receive):")
    print(grid(current))


if __name__ == "__main__":
    main()
