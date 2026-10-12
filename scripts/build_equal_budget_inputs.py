import json
from pathlib import Path


SHOTS = [5, 10, 25, 50, 100, 250, 500, 700]
SAMPLES_PER_CLASS = 700

root = (
    Path.home()
    / "projects/AIDE/dataset/adaptation/recraft_v3_seed42"
)

source_root = root / "low_shot"
output_root = root / "low_shot_equal_budget"


def make_class(source_dir, output_dir):
    sources = sorted(path for path in source_dir.iterdir() if path.is_file())

    if not sources:
        raise RuntimeError(f"No images found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(SAMPLES_PER_CLASS):
        source = sources[index % len(sources)].resolve()
        destination = output_dir / f"{index:04d}_{source.name}"

        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"Unexpected existing link: {destination}")
        else:
            destination.symlink_to(source)

    links = list(output_dir.iterdir())

    if len(links) != SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Expected {SAMPLES_PER_CLASS} links in {output_dir}, "
            f"found {len(links)}"
        )


for shot in SHOTS:
    source = source_root / f"shot_{shot:04d}" / "RecraftV3"
    output = output_root / f"shot_{shot:04d}" / "RecraftV3"

    make_class(source / "0_real", output / "0_real")
    make_class(source / "1_fake", output / "1_fake")

    broken = sum(1 for path in output.rglob("*") if path.is_symlink() and not path.exists())

    print(
        f"shot_{shot:04d}: unique_per_class={shot}, "
        f"training_samples=1400, broken={broken}"
    )

protocol = {
    "name": "recraft_v3_equal_training_budget",
    "shots_per_class": SHOTS,
    "expanded_samples_per_class": SAMPLES_PER_CLASS,
    "total_samples_per_setting": SAMPLES_PER_CLASS * 2,
    "batch_size": 2,
    "epochs": 1,
    "optimizer_steps_per_setting": SAMPLES_PER_CLASS,
    "method": "Cycle selected images and apply AIDE random training augmentations",
}

(output_root / "protocol.json").write_text(
    json.dumps(protocol, indent=2) + "\n"
)

print("Protocol:", output_root / "protocol.json")