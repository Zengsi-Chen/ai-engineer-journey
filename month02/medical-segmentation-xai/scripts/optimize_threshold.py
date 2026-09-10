from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from medseg.data.preprocessing import resize_mask
from medseg.inference.segmenter import Segmenter
from medseg.models.unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "best_model.pt"
)

VAL_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "val.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kvasir-seg"
    / "images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kvasir-seg"
    / "masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
)

IMAGE_SIZE = (256, 256)

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


def load_model(checkpoint_path: Path) -> UNet:
    """Load the Day 34 best checkpoint."""

    model = UNet(
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


def load_ground_truth(mask_path: Path) -> np.ndarray:
    """Load and resize a segmentation mask."""

    mask = Image.open(mask_path).convert("L")

    mask = resize_mask(
        mask,
        IMAGE_SIZE,
    )

    array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return (array > 0).astype(np.uint8)


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Calculate segmentation metrics."""

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = np.logical_and(
        prediction,
        target,
    ).sum()

    false_positive = np.logical_and(
        prediction,
        ~target,
    ).sum()

    false_negative = np.logical_and(
        ~prediction,
        target,
    ).sum()

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    dice = (
        1.0
        if dice_denominator == 0
        else 2 * true_positive / dice_denominator
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = (
        1.0
        if union == 0
        else true_positive / union
    )

    precision_denominator = (
        true_positive
        + false_positive
    )

    precision = (
        0.0
        if precision_denominator == 0
        else true_positive / precision_denominator
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    recall = (
        0.0
        if recall_denominator == 0
        else true_positive / recall_denominator
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def load_validation_dataframe() -> pd.DataFrame:
    """Load validation split."""

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {VAL_CSV}"
        )

    df = pd.read_csv(VAL_CSV)

    print(f"Validation CSV columns: {list(df.columns)}")
    print(f"Validation samples: {len(df)}")

    return df


def resolve_image_name(row: pd.Series) -> str:
    """
    Extract image filename from the validation CSV.

    The split file may use different column names,
    so support common formats.
    """

    possible_columns = [
        "image",
        "image_name",
        "filename",
        "file_name",
        "image_path",
    ]

    for column in possible_columns:

        if column in row.index:

            value = str(row[column])

            return Path(value).name

    raise KeyError(
        "Could not find image filename column "
        f"in validation CSV. Available columns: "
        f"{list(row.index)}"
    )


def main():

    print("=" * 70)
    print("Day 35 Validation Set Threshold Optimization")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    df = load_validation_dataframe()

    model = load_model(
        CHECKPOINT_PATH
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=0.5,
        image_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Step 1:
    # Run inference ONCE per validation image.
    # --------------------------------------------------------

    validation_results = []

    print()
    print("Running validation inference...")
    print()

    for index, row in df.iterrows():

        image_name = resolve_image_name(row)

        image_path = (
            IMAGE_DIR
            / image_name
        )

        mask_path = (
            MASK_DIR
            / image_name
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        # IMPORTANT:
        # Probability map is calculated only once.
        probability_map = (
            segmenter.predict_proba(image)
        )

        target = load_ground_truth(
            mask_path
        )

        validation_results.append(
            {
                "image_name": image_name,
                "probability_map": probability_map,
                "target": target,
            }
        )

        if (index + 1) % 25 == 0:
            print(
                f"Processed "
                f"{index + 1}/{len(df)}"
            )

    print(
        f"Processed "
        f"{len(validation_results)}/{len(df)}"
    )

    # --------------------------------------------------------
    # Step 2:
    # Sweep thresholds.
    # --------------------------------------------------------

    print()
    print(
        f"{'Threshold':>10} "
        f"{'Dice':>10} "
        f"{'IoU':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'FG Ratio':>12}"
    )

    print("-" * 70)

    results = []

    for threshold in THRESHOLDS:

        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        foreground_ratios = []

        for item in validation_results:

            probability_map = (
                item["probability_map"]
            )

            target = item["target"]

            prediction = (
                probability_map >= threshold
            ).astype(np.uint8)

            metrics = calculate_metrics(
                prediction,
                target,
            )

            dice_scores.append(
                metrics["dice"]
            )

            iou_scores.append(
                metrics["iou"]
            )

            precision_scores.append(
                metrics["precision"]
            )

            recall_scores.append(
                metrics["recall"]
            )

            foreground_ratios.append(
                prediction.mean()
            )

        result = {
            "threshold": threshold,
            "dice": float(
                np.mean(dice_scores)
            ),
            "iou": float(
                np.mean(iou_scores)
            ),
            "precision": float(
                np.mean(precision_scores)
            ),
            "recall": float(
                np.mean(recall_scores)
            ),
            "foreground_ratio": float(
                np.mean(foreground_ratios)
            ),
        }

        results.append(result)

        print(
            f"{threshold:10.2f} "
            f"{result['dice']:10.4f} "
            f"{result['iou']:10.4f} "
            f"{result['precision']:12.4f} "
            f"{result['recall']:10.4f} "
            f"{result['foreground_ratio']:11.4%}"
        )

    # --------------------------------------------------------
    # Step 3:
    # Find best threshold based on mean Dice.
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda x: x["dice"],
    )

    print()
    print("=" * 70)
    print("GLOBAL BEST VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold:  {best['threshold']:.2f}"
    )

    print(
        f"Mean Dice:  {best['dice']:.4f}"
    )

    print(
        f"Mean IoU:   {best['iou']:.4f}"
    )

    print(
        f"Precision:   {best['precision']:.4f}"
    )

    print(
        f"Recall:      {best['recall']:.4f}"
    )

    print(
        f"FG Ratio:    "
        f"{best['foreground_ratio']:.4%}"
    )

    # --------------------------------------------------------
    # Step 4:
    # Save optimization results.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    csv_path = (
        OUTPUT_DIR
        / "day35_threshold_optimization.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        f"Results saved to: {csv_path}"
    )

    # --------------------------------------------------------
    # Step 5:
    # Plot threshold vs Dice.
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results_df["threshold"],
        results_df["dice"],
        marker="o",
        label="Mean Dice",
    )

    plt.plot(
        results_df["threshold"],
        results_df["iou"],
        marker="o",
        label="Mean IoU",
    )

    plt.axvline(
        best["threshold"],
        linestyle="--",
        label=(
            f"Best = {best['threshold']:.2f}"
        ),
    )

    plt.xlabel(
        "Segmentation Threshold"
    )

    plt.ylabel(
        "Metric"
    )

    plt.title(
        "Validation Threshold Optimization"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    figure_path = (
        FIGURE_DIR
        / "day35_threshold_optimization.png"
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Figure saved to: {figure_path}"
    )


if __name__ == "__main__":
    main()