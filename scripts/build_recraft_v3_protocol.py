#!/usr/bin/env python3
"""Build a leakage-controlled Recraft-v3 adaptation protocol for AIDE.

The script creates symbolic links, not image copies. It uses Recraft-v3 as
the adaptation source and reserves all other fake generators for an unseen-
generator evaluation. Exact CSV manifests and a JSON protocol summary are
written beside the linked dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path


SEED = 42
SOURCE_GENERATOR = "recraft-v3"
SOURCE_SPLITS = {"train": 700, "val": 150, "source_test": 150}


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=home / "projects/AIDE/dataset/openfake_v1/metadata.csv",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=home / "projects/AIDE/dataset/openfake_v1/OpenFake",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=home / "projects/AIDE/dataset/adaptation/recraft_v3_seed42",
    )
    return parser.parse_args()


def normalized_label(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"0", "real"}:
        return "real"
    if value in {"1", "fake"}:
        return "fake"
    raise ValueError(f"Unsupported label: {value!r}")


def resolve_image(row: dict[str, str], image_root: Path) -> Path:
    original = Path(row["file_path"]).expanduser()
    if original.is_file():
        return original.resolve()

    class_dir = "0_real" if row["label"] == "real" else "1_fake"
    fallback = image_root / class_dir / original.name
    if fallback.is_file():
        return fallback.resolve()

    raise FileNotFoundError(f"Image not found for metadata row {row.get('index')}: {original}")


def load_rows(metadata: Path, image_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"index", "file_path", "label", "model"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

        for row in reader:
            row = dict(row)
            row["label"] = normalized_label(row["label"])
            source = resolve_image(row, image_root)
            row["source_path"] = str(source)
            row["source_relative_path"] = str(source.relative_to(image_root))
            rows.append(row)

    return rows


def link_rows(
    rows: list[dict[str, str]],
    destination: Path,
    split_name: str,
) -> list[dict[str, str]]:
    linked: list[dict[str, str]] = []
    for row in rows:
        class_dir = "0_real" if row["label"] == "real" else "1_fake"
        target_dir = destination / class_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        source = Path(row["source_path"])
        target = target_dir / source.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing path: {target}")
        os.symlink(source, target)

        record = dict(row)
        record["protocol_split"] = split_name
        record["linked_path"] = str(target.absolute())
        linked.append(record)
    return linked


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["index", "file_path", "label", "model"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata = args.metadata.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser()

    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}\n"
            "Nothing was changed. Use a new versioned output path if rebuilding."
        )

    rows = load_rows(metadata, image_root)
    real_rows = [row for row in rows if row["label"] == "real"]
    source_fake = [
        row
        for row in rows
        if row["label"] == "fake" and row["model"].strip().lower() == SOURCE_GENERATOR
    ]
    unseen_fake = [
        row
        for row in rows
        if row["label"] == "fake" and row["model"].strip().lower() != SOURCE_GENERATOR
    ]

    expected_source = sum(SOURCE_SPLITS.values())
    if len(source_fake) != expected_source:
        raise ValueError(
            f"Expected {expected_source} {SOURCE_GENERATOR} images, found {len(source_fake)}"
        )
    if len(real_rows) < expected_source:
        raise ValueError(f"Need at least {expected_source} real images, found {len(real_rows)}")

    rng = random.Random(SEED)
    rng.shuffle(source_fake)
    rng.shuffle(real_rows)

    manifests: dict[str, list[dict[str, str]]] = {}
    fake_offset = 0
    real_offset = 0
    for split, count in SOURCE_SPLITS.items():
        fake_part = source_fake[fake_offset : fake_offset + count]
        real_part = real_rows[real_offset : real_offset + count]
        fake_offset += count
        real_offset += count

        if split == "source_test":
            destination = output_root / "source_test" / "RecraftV3"
        else:
            destination = output_root / split

        split_rows = link_rows(real_part + fake_part, destination, split)
        manifests[split] = split_rows

    remaining_real = real_rows[real_offset:]
    cross_destination = output_root / "cross_generator_test" / "OpenFakeUnseen"
    manifests["cross_generator_test"] = link_rows(
        remaining_real + unseen_fake,
        cross_destination,
        "cross_generator_test",
    )

    for split, split_rows in manifests.items():
        write_manifest(output_root / "manifests" / f"{split}.csv", split_rows)

    summary = {
        "protocol": "recraft_v3_single_source_adaptation_v1",
        "seed": SEED,
        "source_generator": SOURCE_GENERATOR,
        "metadata": str(metadata),
        "image_root": str(image_root),
        "output_root": str(output_root.resolve()),
        "uses_symbolic_links": True,
        "splits": {},
    }
    for split, split_rows in manifests.items():
        counts = Counter(row["label"] for row in split_rows)
        summary["splits"][split] = {
            "total": len(split_rows),
            "real": counts["real"],
            "fake": counts["fake"],
        }

    (output_root / "protocol.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nProtocol created successfully. No image bytes were copied.")


if __name__ == "__main__":
    main()
