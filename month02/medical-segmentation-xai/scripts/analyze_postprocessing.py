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

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
)

IMAGE_SIZE = (256, 256)

# ============================================================
# Day 40 locked operating point
# ============================================================

LOCKED_THRESHOLD = 0.55

# ============================================================
# Day 41 candidate post-processing settings
# ============================================================

MIN_COMPONENT_AREAS = [
    0,
    50,
    100,
    200,
    500,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate connected-component post-processing "
            "on the validation set."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for experiment outputs.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=LOCKED_THRESHOLD,
        help="Locked segmentation threshold.",
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
    """Calculate per-image segmentation metrics."""

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
    """Resolve image and mask paths from split CSV."""

    if "image_path" not in row.index:
        raise KeyError(
            "Validation CSV does not contain "
            "'image_path'. "
            f"Available columns: {list(row.index)}"
        )

    if "mask_path" not in row.index:
        raise KeyError(
            "Validation CSV does not contain "
            "'mask_path'. "
            f"Available columns: {list(row.index)}"
        )

    image_path = Path(
        str(row["image_path"])
    )

    mask_path = Path(
        str(row["mask_path"])
    )

    return image_path, mask_path


def summarize_results(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-image results by min component area."""

    summary = (
        results_df
        .groupby("min_component_area")
        [
            [
                "dice",
                "iou",
                "precision",
                "recall",
                "foreground_ratio",
                "target_foreground_ratio",
            ]
        ]
        .mean()
        .reset_index()
    )

    return summary


def main():

    args = parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    locked_threshold = args.threshold

    print("=" * 70)
    print("Day 41 — Post-processing Validation Experiment")
    print("=" * 70)

    print()
    print(
        "Locked threshold:",
        locked_threshold,
    )

    print(
        "Threshold source:",
        "Day 40 validation operating point",
    )

    print(
        "Candidate min component areas:",
        MIN_COMPONENT_AREAS,
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

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: "
            f"{VAL_CSV}"
        )

    # --------------------------------------------------------
    # Load validation split
    # --------------------------------------------------------

    val_df = pd.read_csv(
        VAL_CSV
    )

    print(
        "Validation CSV columns:",
        list(val_df.columns),
    )

    print(
        "Validation samples:",
        len(val_df),
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
        threshold=locked_threshold,
        image_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Evaluate every validation image
    # --------------------------------------------------------

    results = []

    print()
    print(
        "Running validation inference..."
    )
    print()

    for index, row in val_df.iterrows():

        image_path, mask_path = resolve_paths(
            row
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

        # ----------------------------------------------------
        # Important:
        # Segmenter applies the LOCKED threshold = 0.55.
        # We do NOT change threshold for Day 41.
        # ----------------------------------------------------

        raw_prediction = segmenter.predict(
            image
        )

        # ----------------------------------------------------
        # Test every post-processing setting.
        # ----------------------------------------------------

        for min_area in MIN_COMPONENT_AREAS:

            processed_prediction = (
                remove_small_components(
                    raw_prediction,
                    min_area=min_area,
                )
            )

            metrics = calculate_metrics(
                processed_prediction,
                target,
            )

            results.append(
                {
                    "image_name": image_name,
                    "min_component_area": min_area,
                    **metrics,
                }
            )

        if (index + 1) % 25 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(val_df)}"
            )

    print(
        f"Processed "
        f"{len(val_df)}/"
        f"{len(val_df)}"
    )

    # --------------------------------------------------------
    # Per-image results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Dataset-level summary
    # --------------------------------------------------------

    summary_df = summarize_results(
        results_df
    )

    # --------------------------------------------------------
    # Select best min_area using VALIDATION DICE only
    # --------------------------------------------------------

    best_row = summary_df.loc[
        summary_df["dice"].idxmax()
    ]

    best_min_area = int(
        best_row["min_component_area"]
    )

    best_dice = float(
        best_row["dice"]
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print(
        "Selected min component area:",
        best_min_area,
    )

    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )

    print()
    print(
        "Selection rule:"
    )

    print(
        "Choose min_component_area "
        "with highest mean validation Dice."
    )

    print(
        "Test set was not used "
        "for parameter selection."
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_path = (
        output_dir
        / "day41_postprocessing_per_image.csv"
    )

    summary_path = (
        output_dir
        / "day41_postprocessing_validation.csv"
    )

    json_path = (
        output_dir
        / "day41_postprocessing_validation.json"
    )

    results_df.to_csv(
        per_image_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    experiment_summary = {
        "experiment": (
            "Day 41 Connected Component "
            "Post-processing"
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
        "validation_samples": int(
            len(val_df)
        ),
        "locked_threshold": (
            locked_threshold
        ),
        "threshold_source": (
            "Day 40 validation"
        ),
        "candidate_min_component_areas": (
            MIN_COMPONENT_AREAS
        ),
        "selection_metric": (
            "mean_validation_dice"
        ),
        "selected_min_component_area": (
            best_min_area
        ),
        "best_validation_dice": (
            best_dice
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_summary,
            file,
            indent=2,
        )

    print()
    print(
        "Per-image results saved to:"
    )
    print(per_image_path)

    print()
    print(
        "Validation summary saved to:"
    )
    print(summary_path)

    print()
    print(
        "Experiment summary saved to:"
    )
    print(json_path)


if __name__ == "__main__":
    main()