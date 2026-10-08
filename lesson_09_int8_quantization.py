"""Lesson 09: small, dependency-free int8 quantization experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable

INT8_LIMIT = 127


def round_half_toward_positive_infinity(value: float) -> int:
    """Round to nearest; an exact half goes toward positive infinity."""
    return math.floor(value + 0.5)


def quantize(value: float, scale: float) -> tuple[int, bool]:
    """Return the symmetric int8 code and whether the input was clipped."""
    if scale <= 0:
        raise ValueError("scale must be positive")

    unclamped = round_half_toward_positive_infinity(value / scale)
    clipped = unclamped < -INT8_LIMIT or unclamped > INT8_LIMIT
    code = max(-INT8_LIMIT, min(INT8_LIMIT, unclamped))
    return code, clipped


def automatic_scale(values: Iterable[float]) -> float:
    values = list(values)
    maximum = max((abs(value) for value in values), default=0.0)
    return maximum / INT8_LIMIT if maximum else 1.0


def evaluate(values: list[float], scale: float) -> tuple[float, float, int]:
    errors: list[float] = []
    clipped_count = 0
    for value in values:
        code, clipped = quantize(value, scale)
        reconstructed = code * scale
        errors.append(abs(reconstructed - value))
        clipped_count += int(clipped)

    mean_absolute_error = sum(errors) / len(errors)
    return mean_absolute_error, max(errors), clipped_count


def print_vector_experiment() -> None:
    values = [-1.0, -0.31, 0.0, 0.26, 0.78, 1.4]
    scale = automatic_scale(values)

    print("EXPERIMENT 1: QUANTIZE AND RECONSTRUCT ONE VECTOR")
    print(f"automatic scale = {scale:.8f} (= 1.4 / 127)")
    print(f"{'original':>10} {'int8':>7} {'reconstructed':>15} {'error':>11}")

    for value in values:
        code, _ = quantize(value, scale)
        reconstructed = code * scale
        error = reconstructed - value
        print(f"{value:10.5f} {code:7d} {reconstructed:15.5f} {error:11.5f}")

    mae, maximum_error, clipped = evaluate(values, scale)
    print(f"mean absolute error = {mae:.6f}")
    print(f"maximum absolute error = {maximum_error:.6f}")
    print(f"clipped values = {clipped}\n")


def print_scale_experiment() -> None:
    values = [-1.0, -0.31, 0.0, 0.26, 0.78, 1.4]
    scales = [0.005, automatic_scale(values), 0.05]

    print("EXPERIMENT 2: SCALE TRADES RESOLUTION FOR RANGE")
    print(f"{'scale':>10} {'range':>10} {'MAE':>12} {'max error':>12} {'clipped':>9}")
    for scale in scales:
        mae, maximum_error, clipped = evaluate(values, scale)
        representable_range = INT8_LIMIT * scale
        print(
            f"{scale:10.6f} {representable_range:10.4f} "
            f"{mae:12.6f} {maximum_error:12.6f} {clipped:9d}"
        )
    print()


def print_channel_experiment() -> None:
    channels = {
        "small": [0.01, -0.02, 0.03, -0.04],
        "large": [10.0, -8.0, 6.0, -4.0],
    }
    all_values = [value for values in channels.values() for value in values]
    tensor_scale = automatic_scale(all_values)

    print("EXPERIMENT 3: ONE TENSOR SCALE VS ONE SCALE PER CHANNEL")
    print(f"one tensor-wide scale = {tensor_scale:.8f}")
    print(f"{'channel':>9} {'scheme':>13} {'scale':>12} {'MAE':>12} {'codes':>24}")

    for name, values in channels.items():
        for scheme, scale in (
            ("per tensor", tensor_scale),
            ("per channel", automatic_scale(values)),
        ):
            codes = [quantize(value, scale)[0] for value in values]
            mae, _, _ = evaluate(values, scale)
            print(f"{name:>9} {scheme:>13} {scale:12.8f} {mae:12.8f} {str(codes):>24}")


def main() -> None:
    print_vector_experiment()
    print_scale_experiment()
    print_channel_experiment()


if __name__ == "__main__":
    main()
