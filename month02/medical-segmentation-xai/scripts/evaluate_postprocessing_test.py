from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from medseg.data.preprocessing import resize_mask
from medseg.inference.segmenter import Segmenter
from medseg.models.unet import UNet
from medseg.postprocessing.connected_components import (
    remove_small_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT_PATH = (
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

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
)

IMAGE_SIZE = (256, 256)

# ============================================================
# LOCKED Day 40 + Day 41 configuration
# ============================================================

LOCKED_THRESHOLD = 0.55
LOCKED_MIN_COMPONENT_AREA = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate locked post-processing configuration "
            "on the test set."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
) -> UNet:
    """Load the existing best U-Net checkpoint."""

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
    """Load and resize a ground-truth segmentation mask."""

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


def resolve_paths(
    row: pd.Series,
) -> tuple[Path, Path]:
    """Resolve image and mask paths from test CSV."""

    if "image_path" not in row.index:
        raise KeyError(
            "Test CSV does not contain "
            "'image_path'."
        )

    if "mask_path" not in row.index:
        raise KeyError(
            "Test CSV does not contain "
            "'mask_path'."
        )

    image_path = Path(
        str(row["image_path"])
    )

    mask_path = Path(
        str(row["mask_path"])
    )

    return image_path, mask_path


def main():

    args = parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir

    print("=" * 70)
    print(
        "Day 41 — Locked Post-processing Test Evaluation"
    )
    print("=" * 70)

    print()
    print(
        "LOCKED threshold:",
        f"{LOCKED_THRESHOLD:.2f}",
    )

    print(
        "LOCKED min component area:",
        LOCKED_MIN_COMPONENT_AREA,
    )

    print()
    print(
        "Threshold source:",
        "Day 40 validation",
    )

    print(
        "Post-processing source:",
        "Day 41 validation",
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No parameter optimization is performed "
        "on the test set."
    )

    print()

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
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
        "Test samples:",
        len(test_df),
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        checkpoint_path
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=LOCKED_THRESHOLD,
        image_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Evaluate locked configuration
    # --------------------------------------------------------

    results = []

    print()
    print(
        "Running locked test inference..."
    )
    print()

    for index, row in test_df.iterrows():

        image_path, mask_path = (
            resolve_paths(row)
        )

        image_name = image_path.name

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

        raw_prediction = segmenter.predict(
            image
        )

        processed_prediction = (
            remove_small_components(
                raw_prediction,
                min_area=LOCKED_MIN_COMPONENT_AREA,
            )
        )

        metrics = calculate_metrics(
            processed_prediction,
            target,
        )

        results.append(
            {
                "image_name": image_name,
                "min_component_area": (
                    LOCKED_MIN_COMPONENT_AREA
                ),
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

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Dataset-level metrics
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

    mean_prediction_fg = results_df[
        "foreground_ratio"
    ].mean()

    mean_target_fg = results_df[
        "target_foreground_ratio"
    ].mean()

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL LOCKED TEST RESULTS")
    print("=" * 70)

    print(
        f"Test samples:        "
        f"{len(results_df)}"
    )

    print(
        f"Locked threshold:    "
        f"{LOCKED_THRESHOLD:.2f}"
    )

    print(
        f"Locked min area:     "
        f"{LOCKED_MIN_COMPONENT_AREA}"
    )

    print()

    print(
        f"Mean Dice:           "
        f"{mean_dice:.4f}"
    )

    print(
        f"Mean IoU:            "
        f"{mean_iou:.4f}"
    )

    print(
        f"Mean Precision:      "
        f"{mean_precision:.4f}"
    )

    print(
        f"Mean Recall:         "
        f"{mean_recall:.4f}"
    )

    print()

    print(
        f"Prediction FG ratio: "
        f"{mean_prediction_fg:.4%}"
    )

    print(
        f"Target FG ratio:     "
        f"{mean_target_fg:.4%}"
    )

    # --------------------------------------------------------
    # Save per-image results
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "day41_postprocessing_test.csv"
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
            "Day 41 Locked Post-processing "
            "Test Evaluation"
        ),
        "model": "UNet",
        "features": [
            16,
            32,
            64,
            128,
        ],
        "checkpoint": str(
            checkpoint_path
        ),
        "image_size": list(
            IMAGE_SIZE
        ),
        "test_samples": int(
            len(results_df)
        ),
        "threshold": LOCKED_THRESHOLD,
        "threshold_source": (
            "Day 40 validation"
        ),
        "min_component_area": (
            LOCKED_MIN_COMPONENT_AREA
        ),
        "min_component_area_source": (
            "Day 41 validation"
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
        output_dir
        / "day41_postprocessing_test.json"
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
        "Per-image results saved to:"
    )
    print(csv_path)

    print()
    print(
        "Summary saved to:"
    )
    print(json_path)


if __name__ == "__main__":
    main()