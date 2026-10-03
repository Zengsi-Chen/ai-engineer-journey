"""
Day 43 — Paired Test Audit.

Compares Baseline BCE+Dice and Boundary-aware models
on the exact same test samples.

No retraining or inference is performed.
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
    / "day43_baseline_test_per_image.csv"
)

BOUNDARY_PATH = (
    REPORT_DIR
    / "day43_boundary_01_test_per_image.csv"
)

OUTPUT_PATH = (
    REPORT_DIR
    / "day43_paired_test_audit.csv"
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
    print("Day 43 — Paired Test Audit")
    print("=" * 70)
    print()

    baseline = pd.read_csv(BASELINE_PATH)
    boundary = pd.read_csv(BOUNDARY_PATH)

    print(
        f"Baseline samples: {len(baseline)}"
    )
    print(
        f"Boundary-aware samples: {len(boundary)}"
    )
    print()

    # ---------------------------------------------------------------
    # Merge by sample_id
    # ---------------------------------------------------------------

    paired = baseline.merge(
        boundary,
        on="sample_id",
        suffixes=("_baseline", "_boundary"),
        how="inner",
    )

    print(
        f"Paired samples: {len(paired)}"
    )

    if len(paired) != len(baseline):
        raise ValueError(
            "Not all baseline samples have matching "
            "boundary-aware samples."
        )

    if len(paired) != len(boundary):
        raise ValueError(
            "Not all boundary-aware samples have matching "
            "baseline samples."
        )

    print()

    # ---------------------------------------------------------------
    # Paired metric differences
    # ---------------------------------------------------------------

    for metric in METRICS:
        paired[f"{metric}_change"] = (
            paired[f"{metric}_boundary"]
            - paired[f"{metric}_baseline"]
        )

    # ---------------------------------------------------------------
    # Save paired table
    # ---------------------------------------------------------------

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
    print("PAIRED TEST RESULTS")
    print("=" * 70)
    print()

    print(
        f"{'Metric':<12}"
        f"{'Baseline':>12}"
        f"{'Boundary':>12}"
        f"{'Change':>12}"
    )

    print("-" * 48)

    for metric in METRICS:

        baseline_mean = (
            paired[f"{metric}_baseline"]
            .mean()
        )

        boundary_mean = (
            paired[f"{metric}_boundary"]
            .mean()
        )

        change = (
            boundary_mean
            - baseline_mean
        )

        print(
            f"{metric:<12}"
            f"{baseline_mean:>12.4f}"
            f"{boundary_mean:>12.4f}"
            f"{change:>12.4f}"
        )

    print()

    # ---------------------------------------------------------------
    # Improvement counts
    # ---------------------------------------------------------------

    print("=" * 70)
    print("PAIRED IMPROVEMENT COUNTS")
    print("=" * 70)
    print()

    for metric in [
        "dice",
        "iou",
        "assd",
        "hausdorff",
    ]:

        change_column = (
            f"{metric}_change"
        )

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
            "assd_boundary",
            "hausdorff_baseline",
            "hausdorff_boundary",
        ]
    )

    print("=" * 70)
    print("BOUNDARY-METRIC PAIRED SUBSET")
    print("=" * 70)
    print()

    print(
        f"Samples with valid boundary metrics "
        f"for BOTH models: {len(boundary_paired)}"
    )

    print()

    for metric in [
        "assd",
        "hausdorff",
    ]:

        change_column = (
            f"{metric}_change"
        )

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
    print("=" * 70)
    print("Audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()