from pathlib import Path

import pandas as pd


CSV_PATH = Path("artifacts/reports/day35_test_evaluation.csv")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print("=" * 80)
    print("Day 36 — Segmentation Error Analysis")
    print("=" * 80)

    print(f"\nTest samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    required_columns = [
        "image_name",
        "dice",
        "iou",
        "precision",
        "recall",
        "foreground_ratio",
        "target_foreground_ratio",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------------------------------------------------------
    # 1. Worst cases
    # ---------------------------------------------------------

    worst = df.sort_values("dice", ascending=True).head(20)

    print("\n" + "=" * 80)
    print("WORST 20 SAMPLES BY DICE")
    print("=" * 80)

    print(
        worst[
            [
                "image_name",
                "dice",
                "iou",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 2. Best cases
    # ---------------------------------------------------------

    best = df.sort_values("dice", ascending=False).head(10)

    print("\n" + "=" * 80)
    print("BEST 10 SAMPLES BY DICE")
    print("=" * 80)

    print(
        best[
            [
                "image_name",
                "dice",
                "iou",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 3. Metric distribution
    # ---------------------------------------------------------

    metrics = ["dice", "iou", "precision", "recall"]

    print("\n" + "=" * 80)
    print("METRIC DISTRIBUTION")
    print("=" * 80)

    print(
        df[metrics]
        .describe()
        .loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        .to_string()
    )

    # ---------------------------------------------------------
    # 4. Prediction vs target foreground
    # ---------------------------------------------------------

    df["fg_ratio_error"] = (
        df["foreground_ratio"] - df["target_foreground_ratio"]
    )

    print("\n" + "=" * 80)
    print("FOREGROUND RATIO ERROR")
    print("=" * 80)

    print(
        df["fg_ratio_error"]
        .describe()
        .loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        .to_string()
    )

    # ---------------------------------------------------------
    # 5. Simple failure-mode classification
    # ---------------------------------------------------------

    def classify(row):
        precision = row["precision"]
        recall = row["recall"]

        if precision < 0.50 and recall >= 0.70:
            return "over_segmentation"

        if precision >= 0.70 and recall < 0.60:
            return "under_segmentation"

        if precision < 0.50 and recall < 0.60:
            return "poor_prediction"

        return "mixed_or_reasonable"

    df["failure_mode"] = df.apply(classify, axis=1)

    print("\n" + "=" * 80)
    print("FAILURE MODE COUNTS")
    print("=" * 80)

    print(
        df["failure_mode"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # 6. Save enriched analysis
    # ---------------------------------------------------------

    output_path = Path(
        "artifacts/reports/day36_error_analysis.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Saved:")
    print(output_path)
    print("=" * 80)


if __name__ == "__main__":
    main()