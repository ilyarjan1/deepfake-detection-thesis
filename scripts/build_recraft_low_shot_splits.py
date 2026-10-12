import argparse
import csv
import json
import random
from pathlib import Path


SHOTS = [5, 10, 25, 50, 100, 250, 500, 700]


def parse_args():
    home = Path.home()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=home
        / "projects/AIDE/dataset/adaptation/recraft_v3_seed42/train",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=home
        / "projects/AIDE/dataset/adaptation/recraft_v3_seed42/low_shot",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def shuffled_files(directory, seed):
    files = sorted(path for path in directory.iterdir() if path.is_file())
    random.Random(seed).shuffle(files)
    return files


def create_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = source.resolve()

    if destination.exists() or destination.is_symlink():
        if destination.resolve() != target:
            raise RuntimeError(f"Unexpected existing link: {destination}")
        return

    destination.symlink_to(target)


def main():
    args = parse_args()

    real_files = shuffled_files(args.source_root / "0_real", args.seed)
    fake_files = shuffled_files(args.source_root / "1_fake", args.seed + 1)

    if len(real_files) != 700 or len(fake_files) != 700:
        raise RuntimeError(
            f"Expected 700 files per class, found "
            f"{len(real_files)} real and {len(fake_files)} fake"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    order_path = args.output_root / "selection_order.csv"
    with order_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "rank", "file_name", "source"])

        for label, files in [("real", real_files), ("fake", fake_files)]:
            for rank, path in enumerate(files, 1):
                writer.writerow([label, rank, path.name, str(path.resolve())])

    for shot in SHOTS:
        dataset_root = args.output_root / f"shot_{shot:04d}" / "RecraftV3"
        real_output = dataset_root / "0_real"
        fake_output = dataset_root / "1_fake"

        for source in real_files[:shot]:
            create_link(source, real_output / source.name)

        for source in fake_files[:shot]:
            create_link(source, fake_output / source.name)

        real_count = len(list(real_output.iterdir()))
        fake_count = len(list(fake_output.iterdir()))

        if real_count != shot or fake_count != shot:
            raise RuntimeError(
                f"shot_{shot:04d}: expected {shot} per class, "
                f"found {real_count} real and {fake_count} fake"
            )

        print(
            f"shot_{shot:04d}: "
            f"real={real_count}, fake={fake_count}, total={real_count + fake_count}"
        )

    protocol = {
        "name": "recraft_v3_low_shot_seed42",
        "source_pool": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "shots_per_class": SHOTS,
        "nested_subsets": True,
        "real_images_per_setting": SHOTS,
        "fake_images_per_setting": SHOTS,
        "validation_split": "unchanged 150 real and 150 fake",
        "source_test_split": "unchanged 150 real and 150 fake",
    }

    protocol_path = args.output_root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    print("Protocol:", protocol_path)
    print("Selection order:", order_path)


if __name__ == "__main__":
    main()