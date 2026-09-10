from pathlib import Path

import json

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

TEST_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "test.csv"
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

IMAGE_SIZE = (256, 256)

# ============================================================
# IMPORTANT:
# This threshold was selected on the validation set.
# Do NOT optimize it using the test set.
# ============================================================

LOCKED_THRESHOLD = 0.55


def load_model(
    checkpoint_path: Path,
) -> UNet:
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


def load_ground_truth(
    mask_path: Path,
) -> np.ndarray:
    """Load and resize a segmentation mask."""

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = resize_mask(
        mask,
        IMAGE_SIZE,
    )

    array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return (
        array > 0
    ).astype(np.uint8)


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
        else (
            2 * true_positive
            / dice_denominator
        )
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = (
        1.0
        if union == 0
        else (
            true_positive
            / union
        )
    )

    precision_denominator = (
        true_positive
        + false_positive
    )

    precision = (
        0.0
        if precision_denominator == 0
        else (
            true_positive
            / precision_denominator
        )
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    recall = (
        0.0
        if recall_denominator == 0
        else (
            true_positive
            / recall_denominator
        )
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "foreground_ratio": float(
            prediction.mean()
        ),
        "target_foreground_ratio": float(
            target.mean()
        ),
    }


def resolve_image_name(
    row: pd.Series,
) -> str:
    """Resolve image filename from split CSV."""

    possible_columns = [
        "image",
        "image_name",
        "filename",
        "file_name",
        "image_path",
    ]

    for column in possible_columns:

        if column in row.index:

            value = str(
                row[column]
            )

            return Path(
                value
            ).name

    raise KeyError(
        "Could not find image filename "
        "column in test CSV. "
        f"Available columns: "
        f"{list(row.index)}"
    )


def main():

    print("=" * 70)
    print("Day 35 Final Test Set Evaluation")
    print("=" * 70)

    print()
    print(
        "LOCKED THRESHOLD:",
        LOCKED_THRESHOLD,
    )

    print(
        "Threshold source: "
        "Validation Set"
    )

    print(
        "Test Set will NOT be used "
        "for threshold optimization."
    )

    print()

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found: "
            f"{TEST_CSV}"
        )

    # --------------------------------------------------------
    # Load test split
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_CSV
    )

    print(
        "Test CSV columns:",
        list(test_df.columns),
    )

    print(
        "Test samples:",
        len(test_df),
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        CHECKPOINT_PATH
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=LOCKED_THRESHOLD,
        image_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Evaluate every test image
    # --------------------------------------------------------

    results = []

    print()
    print(
        "Running final test inference..."
    )
    print()

    for index, row in test_df.iterrows():

        image_name = resolve_image_name(
            row
        )

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
                f"Image not found: "
                f"{image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: "
                f"{mask_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        target = load_ground_truth(
            mask_path
        )

        prediction = segmenter.predict(
            image
        )

        metrics = calculate_metrics(
            prediction,
            target,
        )

        results.append(
            {
                "image_name": image_name,
                **metrics,
            }
        )

        if (index + 1) % 25 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test_df)}"
            )

    print(
        f"Processed "
        f"{len(results)}/"
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Calculate dataset-level averages
    # --------------------------------------------------------

    mean_dice = results_df[
        "dice"
    ].mean()

    mean_iou = results_df[
        "iou"
    ].mean()

    mean_precision = results_df[
        "precision"
    ].mean()

    mean_recall = results_df[
        "recall"
    ].mean()

    mean_prediction_fg = (
        results_df[
            "foreground_ratio"
        ].mean()
    )

    mean_target_fg = (
        results_df[
            "target_foreground_ratio"
        ].mean()
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)

    print(
        f"Test samples:       {len(results_df)}"
    )

    print(
        f"Locked threshold:   "
        f"{LOCKED_THRESHOLD:.2f}"
    )

    print()

    print(
        f"Mean Dice:          "
        f"{mean_dice:.4f}"
    )

    print(
        f"Mean IoU:           "
        f"{mean_iou:.4f}"
    )

    print(
        f"Mean Precision:     "
        f"{mean_precision:.4f}"
    )

    print(
        f"Mean Recall:        "
        f"{mean_recall:.4f}"
    )

    print()

    print(
        f"Prediction FG ratio:"
        f" {mean_prediction_fg:.4%}"
    )

    print(
        f"Target FG ratio:    "
        f" {mean_target_fg:.4%}"
    )

    # --------------------------------------------------------
    # Save per-image results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "day35_test_evaluation.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    summary = {
        "experiment": (
            "Day 35 Final Test Evaluation"
        ),
        "model": "UNet",
        "features": [
            16,
            32,
            64,
            128,
        ],
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "image_size": list(
            IMAGE_SIZE
        ),
        "threshold": LOCKED_THRESHOLD,
        "threshold_source": (
            "validation_set"
        ),
        "test_samples": int(
            len(results_df)
        ),
        "metrics": {
            "mean_dice": float(
                mean_dice
            ),
            "mean_iou": float(
                mean_iou
            ),
            "mean_precision": float(
                mean_precision
            ),
            "mean_recall": float(
                mean_recall
            ),
        },
        "foreground_ratio": {
            "prediction": float(
                mean_prediction_fg
            ),
            "target": float(
                mean_target_fg
            ),
        },
    }

    json_path = (
        OUTPUT_DIR
        / "day35_test_evaluation.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        f"Per-image results saved to:"
    )

    print(
        csv_path
    )

    print()
    print(
        f"Summary saved to:"
    )

    print(
        json_path
    )


if __name__ == "__main__":
    main()