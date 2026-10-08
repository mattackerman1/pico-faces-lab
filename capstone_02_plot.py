"""Render a compact PNG summary of the first quantization damage map."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "device-output" / "capstone-02-quant-damage"


def rows(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def color(value, maximum):
    ratio = max(0.0, min(1.0, value / maximum))
    return (255, int(244 - 150 * ratio), int(224 - 185 * ratio))


def main():
    repairs = rows("oracle-repairs.csv")
    lookup = {row["component"]: float(row["mse_recovery_percent"]) for row in repairs}
    image = Image.new("RGB", (1040, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 22), "Pico Faces DiT quantization damage map", fill="black", font=font)
    draw.text((30, 44), "Color/value = final velocity MSE recovered by repairing that component", fill=(60, 60, 60), font=font)

    left, top, cell_w, cell_h = 155, 90, 100, 58
    draw.text((32, top + 20), "Attention", fill="black", font=font)
    draw.text((32, top + cell_h + 20), "MLP", fill="black", font=font)
    maximum = max(lookup[f"block.{b}.attn"] for b in range(8))
    for b in range(8):
        x = left + b * cell_w
        draw.text((x + 34, top - 20), f"B{b}", fill="black", font=font)
        for row_index, kind in enumerate(("attn", "mlp")):
            value = lookup[f"block.{b}.{kind}"]
            y = top + row_index * cell_h
            draw.rectangle((x, y, x + cell_w - 8, y + cell_h - 8), fill=color(value, maximum), outline=(100, 100, 100))
            draw.text((x + 24, y + 17), f"{value:.1f}%", fill="black", font=font)

    draw.text((30, 232), "Inside the two highest-impact attention blocks", fill="black", font=font)
    detail_names = [
        ("B4 input", "block.4.attn.input"), ("B4 softmax", "block.4.attn.softmax"),
        ("B4 Q pre", "block.4.attn.q_pre"), ("B4 all attention", "block.4.attn"),
        ("B5 input", "block.5.attn.input"), ("B5 softmax", "block.5.attn.softmax"),
        ("B5 Q pre", "block.5.attn.q_pre"), ("B5 all attention", "block.5.attn"),
    ]
    chart_left, chart_top, chart_width = 180, 270, 760
    detail_max = max(lookup[key] for _, key in detail_names)
    for index, (label, key) in enumerate(detail_names):
        value = lookup[key]
        y = chart_top + index * 40
        draw.text((30, y + 7), label, fill="black", font=font)
        width = int(chart_width * max(value, 0) / detail_max)
        draw.rectangle((chart_left, y, chart_left + width, y + 26), fill=color(value, detail_max), outline=(120, 120, 120))
        draw.text((chart_left + width + 10, y + 7), f"{value:.2f}%", fill="black", font=font)

    draw.text((30, 605), "Audit: 20 plain + 20 CFG-4 trajectories; 480 DiT calls; fake-deployment int8 graph", fill=(60, 60, 60), font=font)
    image.save(OUT / "damage-map.png")


if __name__ == "__main__":
    main()
