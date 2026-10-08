"""Fetch a small deterministic FFHQ subset from the original holdout range.

The training pipeline's final 2,000 latent rows correspond to images
69000..69999 and their horizontal flips. The official 2.11 GB image archive
is unnecessary for an initial audit: Hugging Face's dataset API can provide
individual cached images. This script samples 128 of those 1,000 images and
downloads the public feature annotations used by the upstream class-label
script.

Dataset note: FFHQ is CC-BY-NC-SA and individual images retain their original
licenses. This subset is for the noncommercial research experiment only.
"""

from __future__ import annotations

import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "device-output" / "capstone-01-adaln-single" / "validation"
CACHE = OUT / "cache"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=nuwandaa%2Fffhq128&config=default&split=train"
    "&offset={offset}&length=100"
)
LABELS_URL = (
    "https://github.com/DCGM/ffhq-features-dataset/"
    "archive/refs/heads/master.zip"
)
SAMPLE_COUNT = 128
RANDOM_SEED = 20261007


def request_bytes(url: str, attempts: int = 3) -> bytes:
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "pico-faces-validation-audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                import time
                time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def selected_indices() -> list[int]:
    rng = random.Random(RANDOM_SEED)
    return sorted(rng.sample(range(69000, 70000), SAMPLE_COUNT))


def image_urls(indices: list[int]) -> dict[int, str]:
    wanted = set(indices)
    found: dict[int, str] = {}
    pages = sorted({index // 100 * 100 for index in indices})
    for page_number, offset in enumerate(pages, 1):
        print(f"metadata page {page_number}/{len(pages)} (offset {offset})")
        payload = json.loads(request_bytes(ROWS_URL.format(offset=offset)))
        for item in payload["rows"]:
            index = int(item["row_idx"])
            if index in wanted:
                found[index] = item["row"]["image"]["src"]
    missing = wanted.difference(found)
    if missing:
        raise RuntimeError(f"Dataset API omitted indices: {sorted(missing)}")
    return found


def download_images(indices: list[int], urls: dict[int, str]) -> np.ndarray:
    images = np.empty((len(indices), 128, 128, 3), dtype=np.uint8)
    for position, index in enumerate(indices):
        data = request_bytes(urls[index])
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if image.size != (128, 128):
            image = image.resize((128, 128), Image.Resampling.LANCZOS)
        images[position] = np.asarray(image, dtype=np.uint8)
        if (position + 1) % 16 == 0 or position + 1 == len(indices):
            print(f"images {position + 1}/{len(indices)}")
    return images


def load_labels(indices: list[int]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE / "ffhq-features-dataset.zip"
    if not archive_path.exists():
        print("downloading public FFHQ feature annotations")
        archive_path.write_bytes(request_bytes(LABELS_URL))
    print(f"annotation archive: {archive_path.stat().st_size / 1e6:.1f} MB")

    labels = np.full(len(indices), 4, dtype=np.uint8)
    with zipfile.ZipFile(archive_path) as archive:
        members = {
            Path(name).name[:5]: name
            for name in archive.namelist()
            if name.endswith(".json") and Path(name).name[:5].isdigit()
        }
        for position, index in enumerate(indices):
            member = members.get(f"{index:05d}")
            if member is None:
                continue
            try:
                parsed = json.loads(archive.read(member))
                attributes = (
                    parsed[0]["faceAttributes"]
                    if isinstance(parsed, list)
                    else parsed["faceAttributes"]
                )
                gender = 0 if attributes["gender"] == "female" else 2
                smile = 1 if float(attributes["smile"]) >= 0.5 else 0
                labels[position] = gender + smile
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                pass
    return labels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    indices = selected_indices()
    urls = image_urls(indices)
    images = download_images(indices, urls)
    labels = load_labels(indices)

    np.save(OUT / "images.npy", images)
    np.save(OUT / "labels.npy", labels)
    np.save(OUT / "indices.npy", np.asarray(indices, dtype=np.int32))
    manifest = {
        "dataset": "nuwandaa/ffhq128",
        "source_range": [69000, 69999],
        "sample_count": len(indices),
        "random_seed": RANDOM_SEED,
        "indices": indices,
        "class_counts": {
            str(class_index): int((labels == class_index).sum())
            for class_index in range(5)
        },
        "caveat": (
            "Images came through the Hugging Face dataset-server JPEG cache, "
            "not byte-identical PNG members from the full upstream zip."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("class counts:", manifest["class_counts"])
    print(f"saved validation subset to {OUT}")


if __name__ == "__main__":
    main()
