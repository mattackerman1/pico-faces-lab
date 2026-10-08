"""Generate one frame on the Pico and compare it with an exact RGB file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from pico_faces_gui import GenerationRequest, generate_on_pico


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--condition", type=int, default=1)
    parser.add_argument("--cfg", type=int, default=4)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = generate_on_pico(GenerationRequest(
        args.port, args.seed, args.steps, args.condition, args.cfg
    ))
    expected = args.expected.read_bytes()
    exact = result.pixels == expected
    raw_path = args.out / "board.rgb"
    png_path = args.out / "board.png"
    raw_path.write_bytes(result.pixels)
    Image.frombytes(
        "RGB" if result.channels == 3 else "L",
        (result.width, result.height),
        result.pixels,
    ).save(png_path)
    report = {
        "port": args.port, "seed": result.seed, "steps": args.steps,
        "condition": result.condition, "cfg": args.cfg,
        "width": result.width, "height": result.height,
        "channels": result.channels, "generation_ms": result.generation_ms,
        "device_crc32": f"{result.device_crc:08x}",
        "host_crc32": f"{result.local_crc:08x}",
        "transport_crc_matches": result.crc_matches,
        "expected_path": str(args.expected),
        "expected_byte_exact": exact,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not result.crc_matches or not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
