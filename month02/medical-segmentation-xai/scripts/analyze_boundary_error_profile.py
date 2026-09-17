from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "artifacts/reports/day42_boundary_validation_per_image.csv"
)

OUTPUT_CSV = Path(
    "artifacts/reports/day42_boundary_error_profile.csv"
)

OUTPUT_JSON = Path(
    "artifacts/reports/day42_boundary_error_profile.json"
)


def classify_row(row: pd.Series) -> str:
    dice = float(row["dice"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    assd = float(row["assd"])
    hausdorff = float(row["hausdorff"])

    # 1. Good segmentation
    if dice >= 0.75 and assd <= 15.0:
        return "good"

    # 2. Broad boundary mismatch
    if dice < 0.50 and assd >= 30.0:
        return "broad_boundary_mismatch"

    # 3. False-positive dominated
    if precision < 0.50 and recall >= 0.50:
        return "fp_dominated"

    # 4. Localized Hausdorff outlier
    if (
        dice >= 0.75
        and assd <= 15.0
        and hausdorff >= 100.0
    ):
        return "localized_hausdorff_outlier"

    # 5. Everything else
    return "mixed_other"


def main() -> None:
    print("=" * 80)
    print("Day 42 — Boundary Error Profile")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_id",
        "image_name",
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

    df["profile"] = df.apply(
        classify_row,
        axis=1,
    )

    total = len(df)

    summary = (
        df["profile"]
        .value_counts()
        .rename_axis("profile")
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"] / total * 100.0
    )

    summary = summary.sort_values(
        "profile"
    ).reset_index(drop=True)

    print()
    print(f"Validation samples: {total}")
    print()
    print("Boundary Error Profile")
    print("-" * 80)

    for _, row in summary.iterrows():
        print(
            f"{row['profile']:30s}"
            f" {int(row['count']):3d}"
            f" ({row['percentage']:6.2f}%)"
        )

    print()
    print("Mean metrics by profile")
    print("-" * 80)

    profile_metrics = (
        df.groupby("profile")[
            [
                "dice",
                "iou",
                "precision",
                "recall",
                "assd",
                "hausdorff",
            ]
        ]
        .mean()
        .round(4)
    )

    print(profile_metrics.to_string())

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    report = {
        "total_validation_samples": int(total),
        "classification_rules": {
            "good": (
                "dice >= 0.75 and assd <= 15.0"
            ),
            "broad_boundary_mismatch": (
                "dice < 0.50 and assd >= 30.0"
            ),
            "fp_dominated": (
                "precision < 0.50 and recall >= 0.50"
            ),
            "localized_hausdorff_outlier": (
                "dice >= 0.75 and assd <= 15.0 "
                "and hausdorff >= 100.0"
            ),
            "mixed_other": (
                "all remaining samples"
            ),
        },
        "profile_summary": summary.to_dict(
            orient="records"
        ),
        "mean_metrics_by_profile": (
            profile_metrics.reset_index().to_dict(
                orient="records"
            )
        ),
    }

    import json

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
    print("Boundary Error Profile complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()