from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Day 40 — Threshold Robustness & Operating Point Analysis
# ============================================================

INPUT_CSV = Path("artifacts/reports/day35_threshold_optimization.csv")

REPORT_DIR = Path("artifacts/reports")
FIGURE_DIR = Path("artifacts/figures")

OUTPUT_CSV = REPORT_DIR / "day40_threshold_robustness.csv"
OUTPUT_JSON = REPORT_DIR / "day40_threshold_robustness.json"
OUTPUT_FIGURE = FIGURE_DIR / "day40_threshold_robustness.png"


def main():
    print("=" * 80)
    print("Day 40 — Threshold Robustness & Operating Point Analysis")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load Day 35 validation threshold results
    # --------------------------------------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "threshold",
        "dice",
        "iou",
        "precision",
        "recall",
        "foreground_ratio",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.sort_values("threshold").reset_index(drop=True)

    # --------------------------------------------------------
    # 2. Identify validation-optimal threshold
    # --------------------------------------------------------
    best_idx = df["dice"].idxmax()
    best_row = df.loc[best_idx]

    best_threshold = float(best_row["threshold"])
    best_dice = float(best_row["dice"])

    # --------------------------------------------------------
    # 3. Define robustness regions
    # --------------------------------------------------------
    ranges = {
        "0.50-0.60": (0.50, 0.60),
        "0.45-0.65": (0.45, 0.65),
    }

    robustness = {}

    for name, (low, high) in ranges.items():
        subset = df[df["threshold"].between(low, high)]

        robustness[name] = {
            "threshold_min": low,
            "threshold_max": high,
            "dice_min": float(subset["dice"].min()),
            "dice_max": float(subset["dice"].max()),
            "dice_range": float(
                subset["dice"].max() - subset["dice"].min()
            ),
            "dice_drop_from_best": float(
                best_dice - subset["dice"].min()
            ),
        }

    # --------------------------------------------------------
    # 4. Calculate degradation relative to best threshold
    # --------------------------------------------------------
    analysis_df = df.copy()

    analysis_df["dice_drop_from_best"] = (
        best_dice - analysis_df["dice"]
    )

    analysis_df["dice_relative_drop_pct"] = (
        analysis_df["dice_drop_from_best"]
        / best_dice
        * 100.0
    )

    analysis_df["is_within_0.50_0.60"] = (
        analysis_df["threshold"].between(0.50, 0.60)
    )

    analysis_df["is_within_0.45_0.65"] = (
        analysis_df["threshold"].between(0.45, 0.65)
    )

    # --------------------------------------------------------
    # 5. Save CSV
    # --------------------------------------------------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    analysis_df.to_csv(OUTPUT_CSV, index=False)

    # --------------------------------------------------------
    # 6. Save JSON summary
    # --------------------------------------------------------
    summary = {
        "best_threshold": best_threshold,
        "best_validation_dice": best_dice,
        "robustness_regions": robustness,
        "operating_point": {
            "threshold": best_threshold,
            "dice": float(best_row["dice"]),
            "iou": float(best_row["iou"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "foreground_ratio": float(
                best_row["foreground_ratio"]
            ),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --------------------------------------------------------
    # 7. Print summary
    # --------------------------------------------------------
    print()
    print("Validation-optimal operating point")
    print("-" * 80)
    print(f"Threshold        : {best_threshold:.2f}")
    print(f"Dice             : {best_row['dice']:.6f}")
    print(f"IoU              : {best_row['iou']:.6f}")
    print(f"Precision        : {best_row['precision']:.6f}")
    print(f"Recall           : {best_row['recall']:.6f}")
    print(
        f"Foreground ratio : "
        f"{best_row['foreground_ratio']:.6f}"
    )

    print()
    print("Threshold robustness")
    print("-" * 80)

    for name, values in robustness.items():
        print(
            f"{name}: "
            f"Dice range = {values['dice_range']:.6f}, "
            f"max degradation = "
            f"{values['dice_drop_from_best']:.6f}"
        )

    print()
    print("Selected operating points")
    print("-" * 80)

    selected = analysis_df[
        analysis_df["threshold"].isin(
            [0.45, 0.50, 0.55, 0.60, 0.65]
        )
    ]

    print(
        selected[
            [
                "threshold",
                "dice",
                "precision",
                "recall",
                "foreground_ratio",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 8. Plot threshold robustness
    # --------------------------------------------------------
    plt.figure(figsize=(9, 6))

    plt.plot(
        analysis_df["threshold"],
        analysis_df["dice"],
        marker="o",
        label="Dice",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Selected threshold = {best_threshold:.2f}",
    )

    plt.axvspan(
        0.45,
        0.65,
        alpha=0.15,
        label="Robustness region",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Dice")
    plt.title(
        "Day 40 — Threshold Robustness on Validation Set"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close()

    print()
    print("Saved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
    