"""
Day 45 — Paired Validation Audit.

Compares the baseline U-Net and Attention U-Net
on the exact same validation samples.

No retraining or validation optimization is performed.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
)

BASELINE_PATH = (
    REPORT_DIR
    / "day44_boundary_ablation"
    / "baseline"
    / "day44_boundary_validation_per_image.csv"
)

ATTENTION_PATH = (
    REPORT_DIR
    / "day45_attention"
    / "attention"
    / "day45_attention_validation_per_image.csv"
)

OUTPUT_PATH = (
    REPORT_DIR
    / "day45_attention"
    / "day45_paired_validation_audit.csv"
)


METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "assd",
    "hausdorff",
]


def main() -> None:
    print("=" * 70)
    print("Day 45 — Paired Validation Audit")
    print("=" * 70)
    print()

    baseline = pd.read_csv(BASELINE_PATH)
    attention = pd.read_csv(ATTENTION_PATH)

    print(f"Baseline samples: {len(baseline)}")
    print(f"Attention samples: {len(attention)}")
    print()

    # ---------------------------------------------------------------
    # Merge by sample_id
    # ---------------------------------------------------------------

    paired = baseline.merge(
        attention,
        on="sample_id",
        suffixes=("_baseline", "_attention"),
        how="inner",
    )

    print(f"Paired samples: {len(paired)}")

    if len(paired) != len(baseline):
        raise ValueError(
            "Not all baseline samples have matching "
            "Attention samples."
        )

    if len(paired) != len(attention):
        raise ValueError(
            "Not all Attention samples have matching "
            "baseline samples."
        )

    print()

    # ---------------------------------------------------------------
    # Paired metric differences
    # Positive = Attention improved for higher-is-better metrics.
    # Negative = Attention improved for lower-is-better metrics.
    # ---------------------------------------------------------------

    for metric in METRICS:
        paired[f"{metric}_change"] = (
            paired[f"{metric}_attention"]
            - paired[f"{metric}_baseline"]
        )

    # ---------------------------------------------------------------
    # Save paired table
    # ---------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Paired results saved to:\n{OUTPUT_PATH}"
    )
    print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("=" * 70)
    print("PAIRED VALIDATION RESULTS")
    print("=" * 70)
    print()

    print(
        f"{'Metric':<12}"
        f"{'Baseline':>12}"
        f"{'Attention':>12}"
        f"{'Change':>12}"
        f"{'Median Δ':>12}"
    )

    print("-" * 60)

    for metric in METRICS:

        baseline_mean = (
            paired[f"{metric}_baseline"]
            .mean()
        )

        attention_mean = (
            paired[f"{metric}_attention"]
            .mean()
        )

        change = (
            attention_mean
            - baseline_mean
        )

        median_change = (
            paired[f"{metric}_change"]
            .median()
        )

        print(
            f"{metric:<12}"
            f"{baseline_mean:>12.4f}"
            f"{attention_mean:>12.4f}"
            f"{change:>12.4f}"
            f"{median_change:>12.4f}"
        )

    print()

    # ---------------------------------------------------------------
    # Improvement counts
    # ---------------------------------------------------------------

    print("=" * 70)
    print("PAIRED IMPROVEMENT COUNTS")
    print("=" * 70)
    print()

    for metric in METRICS:

        change_column = f"{metric}_change"

        valid = paired[
            change_column
        ].dropna()

        if metric in [
            "assd",
            "hausdorff",
        ]:
            improved = (
                valid < 0
            ).sum()

            worsened = (
                valid > 0
            ).sum()

        else:
            improved = (
                valid > 0
            ).sum()

            worsened = (
                valid < 0
            ).sum()

        unchanged = (
            valid == 0
        ).sum()

        print(
            f"{metric:<10}"
            f" improved={improved:>3}  "
            f"worsened={worsened:>3}  "
            f"unchanged={unchanged:>3}  "
            f"valid={len(valid):>3}"
        )

    print()

    # ---------------------------------------------------------------
    # Boundary metric paired subset
    # ---------------------------------------------------------------

    boundary_paired = paired.dropna(
        subset=[
            "assd_baseline",
            "assd_attention",
            "hausdorff_baseline",
            "hausdorff_attention",
        ]
    )

    print("=" * 70)
    print("BOUNDARY-METRIC PAIRED SUBSET")
    print("=" * 70)
    print()

    print(
        "Samples with valid boundary metrics "
        f"for BOTH models: {len(boundary_paired)}"
    )

    print()

    for metric in [
        "assd",
        "hausdorff",
    ]:

        change_column = f"{metric}_change"

        valid = boundary_paired[
            change_column
        ].dropna()

        improved = (
            valid < 0
        ).sum()

        worsened = (
            valid > 0
        ).sum()

        unchanged = (
            valid == 0
        ).sum()

        print(
            f"{metric:<10}"
            f" improved={improved:>3}  "
            f"worsened={worsened:>3}  "
            f"unchanged={unchanged:>3}  "
            f"valid={len(valid):>3}"
        )

    print()

    # ---------------------------------------------------------------
    # Recall vs Precision trade-off
    # ---------------------------------------------------------------

    print("=" * 70)
    print("PRECISION / RECALL TRADE-OFF")
    print("=" * 70)
    print()

    precision_change = (
        paired["precision_change"]
    )

    recall_change = (
        paired["recall_change"]
    )

    precision_improved = (
        precision_change > 0
    ).sum()

    precision_worsened = (
        precision_change < 0
    ).sum()

    recall_improved = (
        recall_change > 0
    ).sum()

    recall_worsened = (
        recall_change < 0
    ).sum()

    print(
        f"Precision improved: {precision_improved}"
    )
    print(
        f"Precision worsened: {precision_worsened}"
    )
    print(
        f"Recall improved:    {recall_improved}"
    )
    print(
        f"Recall worsened:    {recall_worsened}"
    )

    print()

    # ---------------------------------------------------------------
    # Extreme Precision loss cases
    # ---------------------------------------------------------------

    print("=" * 70)
    print("LARGEST PRECISION LOSSES")
    print("=" * 70)
    print()

    largest_precision_losses = (
        paired[
            [
                "sample_id",
                "image_name_baseline",
                "precision_baseline",
                "precision_attention",
                "precision_change",
                "dice_baseline",
                "dice_attention",
                "recall_baseline",
                "recall_attention",
            ]
        ]
        .sort_values(
            "precision_change"
        )
        .head(10)
    )

    print(
        largest_precision_losses.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("Audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
