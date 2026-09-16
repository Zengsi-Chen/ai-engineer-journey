from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "artifacts" / "day38_full_test_failure_taxonomy.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "day38"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Failure taxonomy CSV not found: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "dice",
        "area_ratio",
        "max_probability",
        "saliency_concentration",
        "failure_type",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def plot_dice_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df["dice"],
        bins=15,
        edgecolor="black",
    )

    ax.axvline(
        df["dice"].mean(),
        linestyle="--",
        linewidth=2,
        label=f"Mean Dice = {df['dice'].mean():.3f}",
    )

    ax.set_title("Test Set Dice Distribution")
    ax.set_xlabel("Dice")
    ax.set_ylabel("Number of samples")
    ax.set_xlim(0, 1)
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIR / "01_dice_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_failure_type_distribution(df):
    counts = df["failure_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Test Set Failure Type Distribution")
    ax.set_xlabel("Failure type")
    ax.set_ylabel("Number of samples")

    for index, value in enumerate(counts.values):
        ax.text(
            index,
            value + 1,
            f"{value} ({value / len(df) * 100:.1f}%)",
            ha="center",
        )

    fig.tight_layout()

    path = OUTPUT_DIR / "02_failure_type_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_dice_vs_area_ratio(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for failure_type in df["failure_type"].unique():
        subset = df[df["failure_type"] == failure_type]

        ax.scatter(
            subset["area_ratio"],
            subset["dice"],
            label=failure_type,
            alpha=0.7,
        )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Perfect area ratio",
    )

    ax.set_title("Dice vs Prediction / Ground-Truth Area Ratio")
    ax.set_xlabel("Area ratio (prediction / ground truth)")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIR / "03_dice_vs_area_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_dice_vs_max_probability(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for failure_type in df["failure_type"].unique():
        subset = df[df["failure_type"] == failure_type]

        ax.scatter(
            subset["max_probability"],
            subset["dice"],
            label=failure_type,
            alpha=0.7,
        )

    ax.set_title("Dice vs Maximum Prediction Probability")
    ax.set_xlabel("Maximum predicted probability")
    ax.set_ylabel("Dice")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIR / "04_dice_vs_max_probability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_saliency_vs_dice(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for failure_type in df["failure_type"].unique():
        subset = df[df["failure_type"] == failure_type]

        ax.scatter(
            subset["saliency_concentration"],
            subset["dice"],
            label=failure_type,
            alpha=0.7,
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.5,
        label="Dice = 0.50",
    )

    ax.set_title("Saliency Concentration vs Dice")
    ax.set_xlabel("Saliency concentration (inside GT / outside GT)")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIR / "05_saliency_concentration_vs_dice.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def main():
    print("=" * 80)
    print("Day 38 — Failure Distribution Visualization")
    print("=" * 80)

    df = load_data()

    print(f"\nLoaded {len(df)} test samples.")
    print(f"Mean Dice: {df['dice'].mean():.4f}")
    print(f"Median Dice: {df['dice'].median():.4f}")

    print("\nGenerating diagnostic plots...")

    paths = [
        plot_dice_distribution(df),
        plot_failure_type_distribution(df),
        plot_dice_vs_area_ratio(df),
        plot_dice_vs_max_probability(df),
        plot_saliency_vs_dice(df),
    ]

    print("\nGenerated files:")

    for path in paths:
        print(f"  {path}")

    print("\n" + "=" * 80)
    print("Visualization complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()