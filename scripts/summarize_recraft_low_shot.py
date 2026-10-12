import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aide-root",
        type=Path,
        default=Path.home() / "projects/AIDE",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path.home()
            / "projects/deepfake-detection-thesis"
            / "experiments/005-aide-recraft-low-shot"
            / "results/seed42_metrics.csv"
        ),
    )
    return parser.parse_args()


def prediction_paths(aide_root):
    paths = {
        0: (
            aide_root
            / "results/recraft_v3_original_source_test"
            / "_RecraftV3_predictions.csv"
        ),
        700: (
            aide_root
            / "results/recraft_v3_head_source_test"
            / "_RecraftV3_predictions.csv"
        ),
    }

    for shot in [5, 10, 25, 50, 100, 250, 500]:
        paths[shot] = (
            aide_root
            / "results/recraft_v3_lowshot"
            / f"shot_{shot:04d}_seed42"
            / "source_test"
            / "_RecraftV3_predictions.csv"
        )

    return paths


def calculate(path, shots):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    image_ids = [Path(row["image_path"]).name for row in rows]
    y_true = np.array([int(row["true_label"]) for row in rows])
    probability = np.array(
        [float(row["fake_probability"]) for row in rows]
    )
    prediction = np.array(
        [int(row["predicted_label"]) for row in rows]
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()

    metrics = {
        "shots_per_class": shots,
        "training_images": 0 if shots == 0 else shots * 2,
        "test_images": len(rows),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, prediction
        ),
        "fake_precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "fake_recall": recall_score(y_true, prediction),
        "real_specificity": tn / (tn + fp),
        "fake_f1": f1_score(y_true, prediction),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(
            y_true, probability
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, set(image_ids)


def main():
    args = parse_args()
    paths = prediction_paths(args.aide_root)

    results = []
    reference_images = None

    for shots in sorted(paths):
        path = paths[shots]

        if not path.exists():
            raise FileNotFoundError(path)

        metrics, image_ids = calculate(path, shots)

        if reference_images is None:
            reference_images = image_ids
        elif image_ids != reference_images:
            raise RuntimeError(
                f"{shots}-shot run used different test images"
            )

        results.append(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(
        f"{'shots':>5}  {'acc':>8}  {'recall':>8}  "
        f"{'spec':>8}  {'auc':>8}  {'ap':>8}  "
        f"{'fp':>4}  {'fn':>4}"
    )

    for row in results:
        print(
            f"{row['shots_per_class']:>5}  "
            f"{row['accuracy']:>8.4f}  "
            f"{row['fake_recall']:>8.4f}  "
            f"{row['real_specificity']:>8.4f}  "
            f"{row['roc_auc']:>8.4f}  "
            f"{row['average_precision']:>8.4f}  "
            f"{row['fp']:>4}  "
            f"{row['fn']:>4}"
        )

    print("\nAll runs used the same held-out 300 images.")
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()