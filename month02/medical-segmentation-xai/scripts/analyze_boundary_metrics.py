"""
Day 42 — Boundary-aware validation analysis.

This script evaluates segmentation quality and boundary quality on the
validation set using the locked segmentation threshold from Day 40.

No model retraining or parameter optimization is performed.

Metrics:
- Dice
- IoU
- Precision
- Recall
- ASSD
- Hausdorff distance
"""

from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
import torch

from medseg.evaluation.boundary_metrics import (
    average_symmetric_surface_distance,
    hausdorff_distance,
)
from medseg.inference.segmenter import Segmenter
from medseg.models.unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "best_model.pt"
)

VAL_CSV_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "val.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
)

LOCKED_THRESHOLD = 0.55
IMAGE_SIZE = 256


def resolve_paths(row: pd.Series) -> tuple[Path, Path]:
    """Read image and mask paths directly from the split manifest."""

    if "image_path" not in row.index:
        raise KeyError(
            "Validation manifest must contain image_path"
        )

    if "mask_path" not in row.index:
        raise KeyError(
            "Validation manifest must contain mask_path"
        )

    return (
        Path(str(row["image_path"])),
        Path(str(row["mask_path"])),
    )


def load_binary_mask(
    mask_path: Path,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Load a binary mask and resize it to the model output size."""

    mask = Image.open(mask_path).convert("L")

    mask = mask.resize(
        (image_size[1], image_size[0]),
        Image.Resampling.NEAREST,
    )

    mask_array = np.array(mask)

    return (mask_array > 0).astype(np.uint8)


def calculate_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calculate Dice, IoU, Precision, and Recall."""

    prediction_binary = prediction > 0
    target_binary = target > 0

    intersection = np.logical_and(
        prediction_binary,
        target_binary,
    ).sum()

    prediction_area = prediction_binary.sum()
    target_area = target_binary.sum()

    union = np.logical_or(
        prediction_binary,
        target_binary,
    ).sum()

    true_positive = intersection

    false_positive = np.logical_and(
        prediction_binary,
        ~target_binary,
    ).sum()

    false_negative = np.logical_and(
        ~prediction_binary,
        target_binary,
    ).sum()

    dice = (
        2.0 * intersection + 1.0
    ) / (
        prediction_area
        + target_area
        + 1.0
    )

    iou = (
        intersection + 1.0
    ) / (
        union + 1.0
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    return (
        float(dice),
        float(iou),
        float(precision),
        float(recall),
    )


def calculate_boundary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    """Calculate ASSD and Hausdorff distance for one image."""

    if not np.any(prediction) or not np.any(target):
        return float("nan"), float("nan")

    assd = average_symmetric_surface_distance(
        prediction,
        target,
    )

    hausdorff = hausdorff_distance(
        prediction,
        target,
    )

    return float(assd), float(hausdorff)


def main() -> None:
    print("=" * 70)
    print("Day 42 — Boundary-aware Validation Analysis")
    print("=" * 70)
    print()

    print(f"Locked threshold: {LOCKED_THRESHOLD}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Validation manifest: {VAL_CSV_PATH}")
    print()

    print(
        "No model retraining or validation/test optimization "
        "is performed."
    )
    print()

    df = pd.read_csv(VAL_CSV_PATH)

    print(f"Validation samples: {len(df)}")
    print()

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=LOCKED_THRESHOLD,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    # ---------------------------------------------------------------
    # Validation inference
    # ---------------------------------------------------------------

    results = []

    for index, row in df.iterrows():

        image_path, mask_path = resolve_paths(row)

        image = Image.open(
            image_path
        ).convert("RGB")

        # Same prediction is used for ALL metrics.
        prediction = segmenter.predict(image)

        target = load_binary_mask(
            mask_path,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        # -----------------------------------------------------------
        # Segmentation metrics
        # -----------------------------------------------------------

        dice, iou, precision, recall = (
            calculate_segmentation_metrics(
                prediction,
                target,
            )
        )

        # -----------------------------------------------------------
        # Boundary metrics
        # -----------------------------------------------------------

        assd, hausdorff = calculate_boundary_metrics(
            prediction,
            target,
        )

        results.append(
            {
                "sample_id": row["sample_id"],
                "image_name": image_path.name,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "assd": assd,
                "hausdorff": hausdorff,
                "prediction_empty": int(
                    not np.any(prediction)
                ),
                "target_empty": int(
                    not np.any(target)
                ),
            }
        )

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(df)}"
            )

    # ---------------------------------------------------------------
    # Build result table
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    valid_boundary_df = results_df.dropna(
        subset=["assd", "hausdorff"]
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = {
        "day": 42,
        "threshold": LOCKED_THRESHOLD,
        "validation_samples": int(
            len(results_df)
        ),
        "valid_boundary_samples": int(
            len(valid_boundary_df)
        ),
        "invalid_boundary_samples": int(
            len(results_df)
            - len(valid_boundary_df)
        ),
        "mean_dice": float(
            results_df["dice"].mean()
        ),
        "mean_iou": float(
            results_df["iou"].mean()
        ),
        "mean_precision": float(
            results_df["precision"].mean()
        ),
        "mean_recall": float(
            results_df["recall"].mean()
        ),
        "mean_assd": float(
            valid_boundary_df["assd"].mean()
        ),
        "median_assd": float(
            valid_boundary_df["assd"].median()
        ),
        "mean_hausdorff": float(
            valid_boundary_df["hausdorff"].mean()
        ),
        "median_hausdorff": float(
            valid_boundary_df["hausdorff"].median()
        ),
    }

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_path = (
        OUTPUT_DIR
        / "day42_boundary_validation_per_image.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "day42_boundary_validation.json"
    )

    results_df.to_csv(
        per_image_path,
        index=False,
    )

    import json

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("DAY 42 VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Validation samples: "
        f"{summary['validation_samples']}"
    )

    print(
        f"Valid boundary samples: "
        f"{summary['valid_boundary_samples']}"
    )

    print(
        f"Invalid boundary samples: "
        f"{summary['invalid_boundary_samples']}"
    )

    print()

    print(
        f"Mean Dice: "
        f"{summary['mean_dice']:.4f}"
    )

    print(
        f"Mean IoU: "
        f"{summary['mean_iou']:.4f}"
    )

    print(
        f"Mean Precision: "
        f"{summary['mean_precision']:.4f}"
    )

    print(
        f"Mean Recall: "
        f"{summary['mean_recall']:.4f}"
    )

    print()

    print(
        f"Mean ASSD: "
        f"{summary['mean_assd']:.4f}"
    )

    print(
        f"Median ASSD: "
        f"{summary['median_assd']:.4f}"
    )

    print(
        f"Mean Hausdorff: "
        f"{summary['mean_hausdorff']:.4f}"
    )

    print(
        f"Median Hausdorff: "
        f"{summary['median_hausdorff']:.4f}"
    )

    print()

    print("Per-image results saved to:")
    print(per_image_path)

    print()

    print("Summary saved to:")
    print(summary_path)


if __name__ == "__main__":
    main()