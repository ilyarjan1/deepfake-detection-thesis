from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOME = Path.home()

OUTPUT = Path(
    "experiments/005-aide-recraft-low-shot/"
    "results/cross_generator_seed42"
)

BASE_TABLE = OUTPUT / "generator_comparison.csv"
MATCHED_TABLE = OUTPUT / "matched_fpr.csv"

ADAPTED_PREDICTIONS = HOME / (
    "projects/AIDE/results/recraft_v3_lowshot/"
    "shot_0025_seed42/cross_generator_test/"
    "_OpenFakeUnseen_predictions.csv"
)

METADATA = HOME / (
    "projects/AIDE/dataset/openfake_v1/metadata.csv"
)


def image_id(value):
    return int(Path(str(value)).stem)


OUTPUT.mkdir(parents=True, exist_ok=True)

base = pd.read_csv(BASE_TABLE)
matched = pd.read_csv(MATCHED_TABLE)
predictions = pd.read_csv(ADAPTED_PREDICTIONS)
metadata = pd.read_csv(METADATA)

matched_threshold = float(
    matched.loc[
        matched["method"] == "recraft_25_shot_matched_FPR",
        "threshold",
    ].iloc[0]
)

predictions["image_id"] = predictions["image_path"].map(image_id)
metadata["image_id"] = metadata["index"].astype(int)

joined = predictions.merge(
    metadata[["image_id", "model"]],
    on="image_id",
    how="left",
    validate="one_to_one",
)

if joined["model"].isna().any():
    missing = int(joined["model"].isna().sum())
    raise RuntimeError(f"{missing} predictions did not match metadata")

fake = joined[joined["true_label"].astype(int) == 1].copy()
fake["adapted_matched_prediction"] = (
    fake["fake_probability"].astype(float) >= matched_threshold
).astype(int)

matched_by_generator = (
    fake.groupby("model")
    .agg(
        adapted_matched_detected=(
            "adapted_matched_prediction",
            "sum",
        ),
        adapted_mean_probability=("fake_probability", "mean"),
        adapted_median_probability=("fake_probability", "median"),
    )
    .reset_index()
)

table = base.merge(
    matched_by_generator,
    on="model",
    how="left",
    validate="one_to_one",
)

integer_columns = [
    "images",
    "original_detected",
    "adapted_detected",
    "adapted_matched_detected",
]

for column in integer_columns:
    table[column] = table[column].astype(int)

table["original_missed"] = (
    table["images"] - table["original_detected"]
)
table["adapted_default_missed"] = (
    table["images"] - table["adapted_detected"]
)
table["adapted_matched_missed"] = (
    table["images"] - table["adapted_matched_detected"]
)

table["adapted_matched_recall"] = (
    table["adapted_matched_detected"] / table["images"]
)

table["default_recall_change"] = (
    table["adapted_recall"] - table["original_recall"]
)

table["matched_recall_change"] = (
    table["adapted_matched_recall"] - table["original_recall"]
)

table["matched_extra_detections"] = (
    table["adapted_matched_detected"]
    - table["original_detected"]
)

table["matched_result"] = np.select(
    [
        table["matched_recall_change"] > 1e-12,
        table["matched_recall_change"] < -1e-12,
    ],
    ["improved", "degraded"],
    default="unchanged",
)

columns = [
    "model",
    "images",
    "original_detected",
    "original_missed",
    "original_recall",
    "adapted_detected",
    "adapted_default_missed",
    "adapted_recall",
    "default_recall_change",
    "adapted_matched_detected",
    "adapted_matched_missed",
    "adapted_matched_recall",
    "matched_recall_change",
    "matched_extra_detections",
    "matched_result",
    "adapted_mean_probability",
    "adapted_median_probability",
]

table = table[columns].sort_values(
    "matched_recall_change",
    ascending=False,
)

csv_path = OUTPUT / "generator_detailed_comparison.csv"
table.to_csv(csv_path, index=False)


# Markdown table
markdown_path = OUTPUT / "generator_detailed_comparison.md"

with markdown_path.open("w", encoding="utf-8") as file:
    file.write("# Per-generator cross-generator comparison\n\n")
    file.write(
        f"Matched-FPR adapted threshold: "
        f"`{matched_threshold:.6f}`. "
        "Both models produce 596 false positives on real images.\n\n"
    )
    file.write(
        "| Generator | Images | Original recall | "
        "Adapted recall (0.5) | Change (0.5) | "
        "Adapted recall (matched FPR) | Matched change | Result |\n"
    )
    file.write(
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )

    for row in table.itertuples(index=False):
        file.write(
            f"| {row.model} | {row.images} | "
            f"{row.original_recall:.1%} | "
            f"{row.adapted_recall:.1%} | "
            f"{row.default_recall_change:+.1%} | "
            f"{row.adapted_matched_recall:.1%} | "
            f"{row.matched_recall_change:+.1%} | "
            f"{row.matched_result} |\n"
        )


# Recall comparison chart
plot_table = table.sort_values(
    "original_recall",
    ascending=True,
).reset_index(drop=True)

y = np.arange(len(plot_table))
height = 0.25
figure_height = max(10, len(plot_table) * 0.46)

fig, ax = plt.subplots(figsize=(13, figure_height))

ax.barh(
    y - height,
    plot_table["original_recall"] * 100,
    height,
    label="Original AIDE (threshold 0.5)",
    color="#4C78A8",
)

ax.barh(
    y,
    plot_table["adapted_recall"] * 100,
    height,
    label="Adapted AIDE (threshold 0.5)",
    color="#F2CF5B",
)

ax.barh(
    y + height,
    plot_table["adapted_matched_recall"] * 100,
    height,
    label=f"Adapted AIDE (matched FPR, {matched_threshold:.3f})",
    color="#59A14F",
)

ax.set_yticks(y)
ax.set_yticklabels(plot_table["model"])
ax.set_xlim(0, 100)
ax.set_xlabel("Fake-image recall (%)")
ax.set_title(
    "AIDE performance by unseen generator\n"
    "Original versus Recraft-v3 25-shot adaptation"
)
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()

recall_plot = OUTPUT / "generator_recall_comparison.png"
fig.savefig(recall_plot, dpi=220, bbox_inches="tight")
plt.close(fig)


# Matched-FPR change chart
delta_table = table.sort_values(
    "matched_recall_change",
    ascending=True,
).reset_index(drop=True)

colors = np.where(
    delta_table["matched_recall_change"] >= 0,
    "#3A923A",
    "#D1495B",
)

fig, ax = plt.subplots(figsize=(12, figure_height))

bars = ax.barh(
    np.arange(len(delta_table)),
    delta_table["matched_recall_change"] * 100,
    color=colors,
)

ax.set_yticks(np.arange(len(delta_table)))
ax.set_yticklabels(delta_table["model"])
ax.axvline(0, color="black", linewidth=0.9)
ax.set_xlabel("Recall change at matched false-positive rate (points)")
ax.set_title(
    "Effect of Recraft-v3 25-shot adaptation by generator\n"
    "Green = improvement; red = degradation"
)
ax.grid(axis="x", alpha=0.25)

for bar, value in zip(
    bars,
    delta_table["matched_recall_change"] * 100,
):
    offset = 0.5 if value >= 0 else -0.5
    alignment = "left" if value >= 0 else "right"

    ax.text(
        value + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{value:+.1f}",
        va="center",
        ha=alignment,
        fontsize=8,
    )

fig.tight_layout()

change_plot = OUTPUT / "generator_recall_change_matched_fpr.png"
fig.savefig(change_plot, dpi=220, bbox_inches="tight")
plt.close(fig)


print(f"Matched threshold: {matched_threshold:.6f}")
print(f"Fake images analyzed: {len(fake)}")

print("\nLargest matched-FPR improvements:")
print(
    table[
        [
            "model",
            "images",
            "original_recall",
            "adapted_matched_recall",
            "matched_recall_change",
        ]
    ].head(10).to_string(index=False)
)

print("\nLargest matched-FPR declines:")
print(
    table[
        [
            "model",
            "images",
            "original_recall",
            "adapted_matched_recall",
            "matched_recall_change",
        ]
    ].tail(10).sort_values(
        "matched_recall_change"
    ).to_string(index=False)
)

print("\nSaved:")
print(csv_path.resolve())
print(markdown_path.resolve())
print(recall_plot.resolve())
print(change_plot.resolve())