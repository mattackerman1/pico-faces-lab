"""Small classifier-free-guidance calculation with velocity vectors."""

from typing import TypeAlias

Vector: TypeAlias = list[float]

V_NULL: Vector = [2.0, -1.0, 0.5, 1.5]
V_COND: Vector = [3.0, -0.5, 0.0, 2.0]
GUIDANCE_WEIGHTS = [0, 1, 4, 6, 8]


def subtract(left: Vector, right: Vector) -> Vector:
    return [a - b for a, b in zip(left, right)]


def guided_velocity(v_null: Vector, v_cond: Vector, weight: float) -> Vector:
    class_direction = subtract(v_cond, v_null)
    return [base + weight * direction for base, direction in zip(v_null, class_direction)]


def formatted(values: Vector) -> str:
    return "[" + ", ".join(f"{value:6.2f}" for value in values) + "]"


def main() -> None:
    class_direction = subtract(V_COND, V_NULL)

    print("Classifier-free-guidance calculator")
    print(f"v_null:          {formatted(V_NULL)}")
    print(f"v_cond:          {formatted(V_COND)}")
    print(f"class direction: {formatted(class_direction)}")
    print()

    for weight in GUIDANCE_WEIGHTS:
        result = guided_velocity(V_NULL, V_COND, weight)
        print(f"w={weight:<2} -> v_guided {formatted(result)}")

    assert guided_velocity([2.0], [3.0], 0) == [2.0]
    assert guided_velocity([2.0], [3.0], 1) == [3.0]
    assert guided_velocity([2.0], [3.0], 4) == [6.0]

    print("\nChecks: w=0 is null, w=1 is conditional, and w=4 gives 6 in the scalar example.")


if __name__ == "__main__":
    main()
