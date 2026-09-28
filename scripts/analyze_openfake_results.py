#!/usr/bin/env python3
"""Analyze AIDE per-image predictions on the frozen OpenFake v1 test split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save_metric_table(metrics, path):
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(path, index=False)


def markdown_table(frame):
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if not pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")

    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def grouped_fake_metrics(frame, column):
    rows = []
    fake = frame.loc[frame["true_label"] == 1].copy()

    for value, group in fake.groupby(column, dropna=False):
        probabilities = group["fake_probability"]
        predicted = group["predicted_label"]
        rows.append(
            {
                column: value,
                "images": len(group),
                "detected_fake": int((predicted == 1).sum()),
                "missed_fake": int((predicted == 0).sum()),
                "fake_recall": float((predicted == 1).mean()),
                "mean_fake_probability": float(probabilities.mean()),
                "median_fake_probability": float(probabilities.median()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["fake_recall", "images"], ascending=[True, False]
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    metadata = pd.read_csv(args.metadata)

    required_predictions = {
        "image_path",
        "true_label",
        "fake_probability",
        "predicted_label",
        "correct",
    }
    required_metadata = {
        "index",
        "label",
        "model",
        "type",
        "release_date",
        "prompt",
    }

    missing_predictions = required_predictions - set(predictions.columns)
    missing_metadata = required_metadata - set(metadata.columns)

    if missing_predictions:
        raise ValueError(f"Missing prediction columns: {sorted(missing_predictions)}")
    if missing_metadata:
        raise ValueError(f"Missing metadata columns: {sorted(missing_metadata)}")

    predictions["index"] = (
        predictions["image_path"].map(lambda value: Path(value).stem).astype(int)
    )
    metadata["index"] = metadata["index"].astype(int)

    if predictions["index"].duplicated().any():
        duplicates = predictions.loc[
            predictions["index"].duplicated(keep=False), "index"
        ].tolist()
        raise ValueError(f"Duplicate prediction indices: {duplicates[:10]}")

    if metadata["index"].duplicated().any():
        duplicates = metadata.loc[
            metadata["index"].duplicated(keep=False), "index"
        ].tolist()
        raise ValueError(f"Duplicate metadata indices: {duplicates[:10]}")

    joined = predictions.merge(
        metadata[
            ["index", "label", "model", "type", "release_date", "prompt"]
        ],
        on="index",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = joined.loc[joined["_merge"] != "both", "index"].tolist()
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} predictions lack metadata; first indices: "
            f"{unmatched[:10]}"
        )

    joined = joined.drop(columns="_merge")
    expected_label = joined["label"].map({"real": 0, "fake": 1})

    if expected_label.isna().any():
        unknown = sorted(joined.loc[expected_label.isna(), "label"].unique())
        raise ValueError(f"Unknown metadata labels: {unknown}")

    mismatch = joined["true_label"].astype(int) != expected_label.astype(int)
    if mismatch.any():
        examples = joined.loc[mismatch, ["index", "true_label", "label"]].head(10)
        raise ValueError(f"Prediction/metadata label mismatch:\n{examples}")

    joined["true_label"] = joined["true_label"].astype(int)
    joined["predicted_label"] = joined["predicted_label"].astype(int)
    joined["correct"] = (
        joined["true_label"] == joined["predicted_label"]
    ).astype(int)
    joined["error_type"] = np.select(
        [
            (joined["true_label"] == 1) & (joined["predicted_label"] == 0),
            (joined["true_label"] == 0) & (joined["predicted_label"] == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )

    y_true = joined["true_label"].to_numpy()
    y_pred = joined["predicted_label"].to_numpy()
    probability = joined["fake_probability"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    overall = {
        "images": len(joined),
        "real_images": int((y_true == 0).sum()),
        "fake_images": int((y_true == 1).sum()),
        "correct": int((y_true == y_pred).sum()),
        "errors": int((y_true != y_pred).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity_real": tn / (tn + fp),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": 0.5,
    }

    class_rows = []
    for label_value, label_name in [(0, "real"), (1, "fake")]:
        group = joined.loc[joined["true_label"] == label_value]
        class_rows.append(
            {
                "class": label_name,
                "images": len(group),
                "correct": int(group["correct"].sum()),
                "errors": int((group["correct"] == 0).sum()),
                "class_accuracy": float(group["correct"].mean()),
                "mean_fake_probability": float(group["fake_probability"].mean()),
                "median_fake_probability": float(
                    group["fake_probability"].median()
                ),
            }
        )

    generator = grouped_fake_metrics(joined, "model")
    generator_type = grouped_fake_metrics(joined, "type")
    release_date = grouped_fake_metrics(joined, "release_date")

    false_negatives = joined.loc[
        joined["error_type"] == "false_negative"
    ].sort_values("fake_probability", ascending=True)

    false_positives = joined.loc[
        joined["error_type"] == "false_positive"
    ].sort_values("fake_probability", ascending=False)

    joined.to_csv(args.output_dir / "joined_predictions.csv", index=False)
    save_metric_table(overall, args.output_dir / "overall_metrics.csv")
    pd.DataFrame(class_rows).to_csv(
        args.output_dir / "class_metrics.csv", index=False
    )
    generator.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    generator_type.to_csv(args.output_dir / "generator_type_metrics.csv", index=False)
    release_date.to_csv(args.output_dir / "release_date_metrics.csv", index=False)
    false_negatives.to_csv(
        args.output_dir / "false_negatives.csv", index=False
    )
    false_positives.to_csv(
        args.output_dir / "false_positives.csv", index=False
    )

    worst = generator.head(10)
    best = generator.sort_values(
        ["fake_recall", "images"], ascending=[False, False]
    ).head(10)

    summary = [
        "# AIDE on OpenFake v1",
        "",
        "## Overall results",
        "",
        f"- Images: {overall['images']:,}",
        f"- Accuracy: {overall['accuracy']:.4%}",
        f"- Balanced accuracy: {overall['balanced_accuracy']:.4%}",
        f"- Fake precision: {overall['precision_fake']:.4%}",
        f"- Fake recall: {overall['recall_fake_sensitivity']:.4%}",
        f"- Real specificity: {overall['specificity_real']:.4%}",
        f"- Fake F1: {overall['f1_fake']:.4%}",
        f"- ROC-AUC: {overall['roc_auc']:.6f}",
        f"- Average precision: {overall['average_precision']:.6f}",
        f"- False positives: {overall['false_positive']:,}",
        f"- False negatives: {overall['false_negative']:,}",
        "",
        "All threshold-based metrics use the pretrained model's unchanged "
        "0.5 decision threshold.",
        "",
        "## Lowest fake recall by generator",
        "",
        markdown_table(worst),
        "",
        "## Highest fake recall by generator",
        "",
        markdown_table(best),
        "",
    ]

    (args.output_dir / "summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("Analysis complete")
    print(f"Joined rows: {len(joined)}")
    print(f"Accuracy: {overall['accuracy']:.6f}")
    print(f"Fake recall: {overall['recall_fake_sensitivity']:.6f}")
    print(f"Real specificity: {overall['specificity_real']:.6f}")
    print(f"Fake precision: {overall['precision_fake']:.6f}")
    print(f"Fake F1: {overall['f1_fake']:.6f}")
    print(f"ROC-AUC: {overall['roc_auc']:.6f}")
    print(f"Average precision: {overall['average_precision']:.6f}")
    print(f"False positives: {overall['false_positive']}")
    print(f"False negatives: {overall['false_negative']}")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
