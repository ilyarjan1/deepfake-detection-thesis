#!/usr/bin/env python3
"""Build local contact sheets for visual inspection of AIDE OpenFake failures."""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_MODELS = [
    "recraft-v3",
    "imagen-4.0",
    "grok-2-image-1212",
    "flux-1.1-pro",
    "midjourney-6",
    "dalle-3",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", type=Path, required=True)
    parser.add_argument("--aide-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser.parse_args()


def resolve_image(aide_root, stored_path):
    path = Path(stored_path)
    return path if path.is_absolute() else aide_root / path


def make_tile(path, probability, outcome, size=(240, 290)):
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    border = "#c62828" if outcome == "missed_fake" else "#2e7d32"

    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((size[0] - 12, size[1] - 54))
            x = (size[0] - image.width) // 2
            y = 6 + (size[1] - 54 - image.height) // 2
            canvas.paste(image, (x, y))
    except Exception as exc:
        draw.text((10, 40), f"Unreadable\n{type(exc).__name__}", fill="black")

    draw.rectangle((1, 1, size[0] - 2, size[1] - 2), outline=border, width=5)
    label = f"{outcome}  p(fake)={probability:.3f}"
    draw.text((8, size[1] - 39), label, fill="black")
    draw.text((8, size[1] - 21), path.name, fill="black")
    return canvas


def contact_sheet(rows, title, aide_root, columns=6):
    tile_width, tile_height = 240, 290
    title_height = 44
    row_count = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, title_height + row_count * tile_height),
        "#eeeeee",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 13), title, fill="black")

    for position, row in enumerate(rows.itertuples(index=False)):
        path = resolve_image(aide_root, row.image_path)
        tile = make_tile(
            path,
            float(row.fake_probability),
            str(row.visual_group),
            (tile_width, tile_height),
        )
        x = (position % columns) * tile_width
        y = title_height + (position // columns) * tile_height
        sheet.paste(tile, (x, y))

    return sheet


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.joined)
    required = {
        "index",
        "image_path",
        "true_label",
        "predicted_label",
        "fake_probability",
        "model",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    manifests = []

    for model in args.models:
        group = data[(data["model"] == model) & (data["true_label"] == 1)].copy()
        if group.empty:
            print(f"Skipping missing model: {model}")
            continue

        missed = (
            group[group["predicted_label"] == 0]
            .nsmallest(args.per_group, "fake_probability")
            .copy()
        )
        detected = (
            group[group["predicted_label"] == 1]
            .nlargest(args.per_group, "fake_probability")
            .copy()
        )

        missed["visual_group"] = "missed_fake"
        detected["visual_group"] = "detected_fake"
        selected = pd.concat([missed, detected], ignore_index=True)

        selected["generator"] = model
        manifests.append(selected)

        sheet = contact_sheet(
            selected,
            (
                f"{model}: {len(missed)} confident misses (red) vs "
                f"{len(detected)} confident detections (green)"
            ),
            args.aide_root,
        )
        output = args.output_dir / f"{model}_failure_gallery.jpg"
        sheet.save(output, quality=92)
        print(f"Saved: {output}")

    if not manifests:
        raise RuntimeError("No requested generators were found")

    manifest = pd.concat(manifests, ignore_index=True)
    manifest.to_csv(args.output_dir / "gallery_manifest.csv", index=False)
    print(f"Selected images: {len(manifest)}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
