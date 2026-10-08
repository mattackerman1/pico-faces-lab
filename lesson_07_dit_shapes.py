"""Trace Pico-Faces DiT shapes and verify patchify/unpatchify exactly."""

from typing import TypeAlias

HEIGHT = 16
WIDTH = 16
CHANNELS = 8
PATCH = 2
TOKEN_WIDTH = 128
FAST_BLOCKS = 8
QUALITY_BLOCKS = 12

Latent: TypeAlias = list[list[list[int]]]
Patch: TypeAlias = list[int]


def make_numbered_latent() -> Latent:
    """Give every location/channel a unique value so movement is traceable."""
    return [
        [[y * 10_000 + x * 100 + channel for channel in range(CHANNELS)] for x in range(WIDTH)]
        for y in range(HEIGHT)
    ]


def patchify(latent: Latent) -> list[Patch]:
    patches: list[Patch] = []
    for patch_y in range(0, HEIGHT, PATCH):
        for patch_x in range(0, WIDTH, PATCH):
            values: Patch = []
            for offset_y in range(PATCH):
                for offset_x in range(PATCH):
                    values.extend(latent[patch_y + offset_y][patch_x + offset_x])
            patches.append(values)
    return patches


def unpatchify(patches: list[Patch]) -> Latent:
    latent: Latent = [
        [[0 for _ in range(CHANNELS)] for _ in range(WIDTH)]
        for _ in range(HEIGHT)
    ]
    patch_index = 0
    for patch_y in range(0, HEIGHT, PATCH):
        for patch_x in range(0, WIDTH, PATCH):
            values = patches[patch_index]
            value_index = 0
            for offset_y in range(PATCH):
                for offset_x in range(PATCH):
                    latent[patch_y + offset_y][patch_x + offset_x] = values[
                        value_index : value_index + CHANNELS
                    ]
                    value_index += CHANNELS
            patch_index += 1
    return latent


def main() -> None:
    latent = make_numbered_latent()
    patches = patchify(latent)
    restored = unpatchify(patches)

    patches_across = WIDTH // PATCH
    patch_values = PATCH * PATCH * CHANNELS
    token_count = len(patches)

    assert token_count == (HEIGHT // PATCH) * (WIDTH // PATCH)
    assert all(len(patch) == patch_values for patch in patches)
    assert restored == latent

    print("Pico-Faces DiT shape trace")
    print(f"latent:             {HEIGHT} x {WIDTH} x {CHANNELS} = {HEIGHT * WIDTH * CHANNELS:,} values")
    print(f"patch grid:         {patches_across} x {patches_across} = {token_count} patches")
    print(f"one raw patch:      {PATCH} x {PATCH} x {CHANNELS} = {patch_values} values")
    print(f"raw patch matrix:   {token_count} x {patch_values} = {token_count * patch_values:,} values")
    print(f"embedded tokens:    {token_count} x {TOKEN_WIDTH} = {token_count * TOKEN_WIDTH:,} working values")
    print(f"fast DiT blocks:    {FAST_BLOCKS}, shape remains {token_count} x {TOKEN_WIDTH}")
    print(f"quality DiT blocks: {QUALITY_BLOCKS}, shape remains {token_count} x {TOKEN_WIDTH}")
    print(f"output patches:     {token_count} x {patch_values} velocity values")
    print(f"velocity tensor:    {HEIGHT} x {WIDTH} x {CHANNELS}")
    print()
    print("Patchify -> unpatchify check: EXACT")
    print("First patch contains all 8 channels from positions:")
    print("  (row 0, col 0), (row 0, col 1), (row 1, col 0), (row 1, col 1)")
    print(f"First patch's 32 traceable values:\n{patches[0]}")


if __name__ == "__main__":
    main()
