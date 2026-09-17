from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "artifacts/reports/day42_boundary_error_profile.csv"
)

OUTPUT_CSV = Path(
    "artifacts/reports/day42_mixed_other_decomposition.csv"
)

OUTPUT_JSON = Path(
    "artifacts/reports/day42_mixed_other_decomposition.json"
)


def classify_mixed_row(row: pd.Series) -> str:
    """
    Decompose samples previously classified as mixed_other.

    Rules are mutually exclusive because they are evaluated
    in a fixed priority order.
    """

    dice = float(row["dice"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    assd = float(row["assd"])
    hausdorff = float(row["hausdorff"])

    # 1. High-quality / near-good
    if dice >= 0.70 and assd <= 15.0:
        return "high_quality"

    # 2. Moderate boundary error
    if dice >= 0.50 and assd > 15.0:
        return "moderate_boundary_error"

    # 3. False-positive heavy
    if precision < 0.60 and recall >= 0.70:
        return "fp_heavy"

    # 4. False-negative / under-segmentation tendency
    if recall < 0.60 and precision >= 0.60:
        return "fn_heavy"

    # 5. Localized Hausdorff outlier
    if hausdorff >= 120.0 and assd < 30.0:
        return "hausdorff_outlier"

    # 6. Severe overlap failure without broad boundary mismatch
    if dice < 0.50 and assd < 30.0:
        return "severe_mixed"

    # 7. Remaining cases
    return "other"


def main() -> None:
    print("=" * 80)
    print("Day 42 — Mixed/Other Failure Decomposition")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_id",
        "image_name",
        "profile",
        "dice",
        "iou",
        "precision",
        "recall",
        "assd",
        "hausdorff",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    mixed_df = df[df["profile"] == "mixed_other"].copy()

    total = len(mixed_df)

    print()
    print(f"Original mixed_other samples: {total}")

    mixed_df["mixed_subtype"] = mixed_df.apply(
        classify_mixed_row,
        axis=1,
    )

    summary = (
        mixed_df["mixed_subtype"]
        .value_counts()
        .rename_axis("mixed_subtype")
        .reset_index(name="count")
    )

    summary["percentage_of_mixed"] = (
        summary["count"] / total * 100.0
    )

    summary["percentage_of_validation"] = (
        summary["count"] / len(df) * 100.0
    )

    summary = summary.sort_values(
        "mixed_subtype"
    ).reset_index(drop=True)

    print()
    print("Mixed/Other Decomposition")
    print("-" * 80)

    for _, row in summary.iterrows():
        print(
            f"{row['mixed_subtype']:30s}"
            f" {int(row['count']):3d}"
            f" ({row['percentage_of_mixed']:6.2f}% of mixed)"
            f" ({row['percentage_of_validation']:6.2f}% of val)"
        )

    print()
    print("Mean metrics by mixed subtype")
    print("-" * 80)

    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
        "assd",
        "hausdorff",
    ]

    profile_metrics = (
        mixed_df.groupby("mixed_subtype")[metric_columns]
        .mean()
        .round(4)
    )

    print(profile_metrics.to_string())

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mixed_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    report = {
        "source_profile": "mixed_other",
        "source_sample_count": int(total),
        "total_validation_samples": int(len(df)),
        "classification_rules": {
            "high_quality": (
                "dice >= 0.70 and assd <= 15.0"
            ),
            "moderate_boundary_error": (
                "dice >= 0.50 and assd > 15.0"
            ),
            "fp_heavy": (
                "precision < 0.60 and recall >= 0.70"
            ),
            "fn_heavy": (
                "recall < 0.60 and precision >= 0.60"
            ),
            "hausdorff_outlier": (
                "hausdorff >= 120.0 and assd < 30.0"
            ),
            "severe_mixed": (
                "dice < 0.50 and assd < 30.0"
            ),
            "other": (
                "all remaining samples"
            ),
        },
        "priority_order": [
            "high_quality",
            "moderate_boundary_error",
            "fp_heavy",
            "fn_heavy",
            "hausdorff_outlier",
            "severe_mixed",
            "other",
        ],
        "summary": summary.to_dict(
            orient="records"
        ),
        "mean_metrics_by_subtype": (
            profile_metrics
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")

    print()
    print("=" * 80)
    print("Mixed/Other decomposition complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()