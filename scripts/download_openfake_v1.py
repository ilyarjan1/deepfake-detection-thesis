import argparse
import os
import sys
import csv
from collections import Counter
from io import BytesIO
from pathlib import Path

from datasets import Image, load_dataset
from PIL import Image as PILImage


EXPECTED_TOTAL = 59658
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def determine_extension(image_bytes, original_path):
    extension = Path(original_path or "").suffix.lower()

    if extension in VALID_EXTENSIONS:
        return extension

    image_format = PILImage.open(BytesIO(image_bytes)).format
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
    }.get(image_format, ".jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="v1.0")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home()
        / "projects/AIDE/dataset/openfake_v1/OpenFake",
    )
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    metadata_path = root.parent / "metadata.csv"

    class_dirs = {
        "real": root / "0_real",
        "fake": root / "1_fake",
    }

    for directory in class_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    completed = set()
    counts = Counter()

    if metadata_path.exists():
        with metadata_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                completed.add(int(row["index"]))
                counts[row["label"]] += 1

    print("Dataset: ComplexDataLab/OpenFake")
    print("Revision:", args.revision)
    print("Split: test")
    print("Previously exported:", len(completed))

    dataset = load_dataset(
        "ComplexDataLab/OpenFake",
        split="test",
        revision=args.revision,
        streaming=True,
    )
    dataset = dataset.cast_column("image", Image(decode=False))

    fields = [
        "index",
        "file_path",
        "label",
        "model",
        "type",
        "release_date",
        "prompt",
    ]

    write_header = (
        not metadata_path.exists()
        or metadata_path.stat().st_size == 0
    )

    with metadata_path.open(
        "a", newline="", encoding="utf-8"
    ) as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=fields)

        if write_header:
            writer.writeheader()

        for index, row in enumerate(dataset):
            if args.limit is not None and index >= args.limit:
                break

            if index in completed:
                continue

            label = str(row["label"]).strip().lower()

            if label not in class_dirs:
                raise ValueError(
                    f"Unexpected label at index {index}: {label!r}"
                )

            image = row["image"]
            image_bytes = image.get("bytes")
            original_path = image.get("path") or ""

            if image_bytes is None:
                image_bytes = Path(original_path).read_bytes()

            extension = determine_extension(
                image_bytes,
                original_path,
            )

            output_path = (
                class_dirs[label] / f"{index:06d}{extension}"
            )

            # Preserve original bytes without recompression.
            output_path.write_bytes(image_bytes)

            writer.writerow(
                {
                    "index": index,
                    "file_path": str(output_path),
                    "label": label,
                    "model": row.get("model", ""),
                    "type": row.get("type", ""),
                    "release_date": row.get("release_date", ""),
                    "prompt": row.get("prompt", ""),
                }
            )

            counts[label] += 1

            if (index + 1) % 100 == 0:
                metadata_file.flush()
                print(
                    f"{index + 1}/{EXPECTED_TOTAL} | "
                    f"real={counts['real']} "
                    f"fake={counts['fake']}"
                )

    print("Export completed.")
    print("Real:", counts["real"])
    print("Fake:", counts["fake"])
    print("Total:", sum(counts.values()))
    print("Images:", root)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
