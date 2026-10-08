"""A small demonstration of RMS normalization and an AdaLN-Zero residual gate."""

from math import sqrt
from typing import TypeAlias

Vector: TypeAlias = list[float]


def rms_norm(values: Vector) -> Vector:
    rms = sqrt(sum(value * value for value in values) / len(values))
    return [value / rms for value in values]


def modulate(normalized: Vector, scale: Vector, shift: Vector) -> Vector:
    return [value * (1.0 + s) + b for value, s, b in zip(normalized, scale, shift)]


def residual_update(original: Vector, proposal: Vector, gate: Vector) -> Vector:
    return [x + g * proposed for x, proposed, g in zip(original, proposal, gate)]


def main() -> None:
    token = [3.0, 4.0]
    normalized = rms_norm(token)

    zero_scale = [0.0, 0.0]
    zero_shift = [0.0, 0.0]
    zero_gate = [0.0, 0.0]
    proposal = [0.8, -0.6]  # stand-in for an attention or MLP result

    adjusted_at_init = modulate(normalized, zero_scale, zero_shift)
    output_at_init = residual_update(token, proposal, zero_gate)

    learned_scale = [0.20, -0.10]
    learned_shift = [0.10, 0.05]
    learned_gate = [0.25, 0.50]
    adjusted_after_training = modulate(normalized, learned_scale, learned_shift)
    output_after_training = residual_update(token, proposal, learned_gate)

    print("AdaLN-Zero toy demonstration")
    print(f"token:                    {token}")
    print(f"RMS-normalized:           {[round(x, 3) for x in normalized]}")
    print(f"adjusted at zero init:    {[round(x, 3) for x in adjusted_at_init]}")
    print(f"branch proposal:          {proposal}")
    print(f"output with zero gate:    {[round(x, 3) for x in output_at_init]}")
    print()
    print(f"adjusted after learning:  {[round(x, 3) for x in adjusted_after_training]}")
    print(f"output with learned gate: {[round(x, 3) for x in output_after_training]}")

    assert output_at_init == token
    print("\nCheck: a zero residual gate makes the new block a pass-through.")


if __name__ == "__main__":
    main()
