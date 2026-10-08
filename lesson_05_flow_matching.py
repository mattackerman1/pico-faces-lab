"""A dependency-free, one-dimensional flow-matching demonstration."""

NOISE = -3.0
CLEAN = 5.0
STEPS = 4


def interpolate(x0: float, x1: float, t: float) -> float:
    """Return the straight path between x0 at t=0 and x1 at t=1."""
    return t * x1 + (1.0 - t) * x0


def perfect_model_velocity(current: float, time: float) -> float:
    """Stand in for a DiT call; the toy's exact velocity is constant."""
    del current, time
    return CLEAN - NOISE


def main() -> None:
    target_velocity = CLEAN - NOISE
    step_size = 1.0 / STEPS
    current = NOISE

    print("Flow-matching toy example")
    print(f"noise x0={NOISE:g}, clean x1={CLEAN:g}, target velocity v={target_velocity:g}")
    print()
    print("step   time   state before   predicted v   change   state after")

    for step in range(1, STEPS + 1):
        time = (step - 1) * step_size
        predicted_velocity = perfect_model_velocity(current, time)
        change = step_size * predicted_velocity
        before = current
        current += change
        print(
            f"{step:>4}   {time:>4.2f}   {before:>12.2f}   "
            f"{predicted_velocity:>11.2f}   {change:>6.2f}   {current:>11.2f}"
        )

    assert abs(current - CLEAN) < 1e-12
    print("\nThe stand-in model was queried four times.")
    print("Because this toy velocity is constant and perfect, it reaches the clean endpoint exactly.")


if __name__ == "__main__":
    main()
