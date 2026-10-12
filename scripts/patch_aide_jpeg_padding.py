#!/usr/bin/env python3
"""Make AIDE's Kornia JPEG augmentation safe for arbitrary image sizes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMPORT_ANCHOR = "import torch\n"
IMPORT_REPLACEMENT = "import torch\nimport torch.nn.functional as F\n"

TRANSFORM_ANCHOR = '''transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    ]
)
'''

TRANSFORM_REPLACEMENT = '''def apply_perturbations_with_jpeg_padding(image):
    """Pad for JPEG's 16-pixel blocks, then restore the original shape."""
    height, width = image.shape[-2:]
    pad_height = (-height) % 16
    pad_width = (-width) % 16

    if pad_height or pad_width:
        image_padded = F.pad(
            image,
            (0, pad_width, 0, pad_height),
            mode='reflect',
        )
    else:
        image_padded = image

    transformed = Perturbations(image_padded)[0]
    return transformed[..., :height, :width]


transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(apply_perturbations_with_jpeg_padding)
    ]
)
'''


def replace_once(text: str, anchor: str, replacement: str, name: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"Expected one {name} anchor, found {count}; nothing was modified."
        )
    return text.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aide-root",
        type=Path,
        default=Path.home() / "projects/AIDE",
    )
    args = parser.parse_args()

    target = args.aide_root.expanduser().resolve() / "data" / "datasets.py"
    backup = target.with_name("datasets.py.before-jpeg-padding")

    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    if "apply_perturbations_with_jpeg_padding" in original:
        raise RuntimeError("JPEG padding patch is already installed; nothing changed.")

    patched = replace_once(original, IMPORT_ANCHOR, IMPORT_REPLACEMENT, "import")
    patched = replace_once(
        patched,
        TRANSFORM_ANCHOR,
        TRANSFORM_REPLACEMENT,
        "transform",
    )
    compile(patched, str(target), "exec")

    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}\nNothing was modified."
        )

    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print("JPEG augmentation now pads to multiples of 16 and crops back.")


if __name__ == "__main__":
    main()
