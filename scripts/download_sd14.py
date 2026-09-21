from collections import Counter
from pathlib import Path

from datasets import Image, load_dataset

print("Downloading GenImage SD v1.4 test dataset...")

ds = load_dataset(
    "nebula/GenImage-arrow",
    "sd14-test",
    split="test",
)

ds = ds.cast_column("image", Image(decode=False))

root = Path("dataset/genimage_sd14/StableDiffusionV14")
folders = {
    0: root / "0_real",
    1: root / "1_fake",
}

for folder in folders.values():
    folder.mkdir(parents=True, exist_ok=True)

counts = Counter()

for index, row in enumerate(ds):
    label = int(row["label"])
    image = row["image"]

    data = image.get("bytes")
    if data is None:
        data = Path(image["path"]).read_bytes()

    original_path = row.get("image_path") or image.get("path") or ""
    extension = Path(original_path).suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        extension = ".png"

    output = folders[label] / f"{index:06d}{extension}"
    output.write_bytes(data)
    counts[label] += 1

print("Finished")
print("Real images:", counts[0])
print("Fake images:", counts[1])
print("Total images:", sum(counts.values()))
print("Saved under:", root.resolve())