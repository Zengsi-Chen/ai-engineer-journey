from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from medseg.models.factory import create_model
from medseg.inference.segmenter import Segmenter
from medseg.evaluation.boundary_metrics import extract_boundary


CSV_PATH = Path(
    "artifacts/reports/day42_boundary_validation_per_image.csv"
)

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/best_model.pt"
)

OUTPUT_DIR = Path(
    "artifacts/figures/day42_boundary_cases"
)

IMAGE_DIR = Path(
    "data/raw/kvasir-seg/images"
)

MASK_DIR = Path(
    "data/raw/kvasir-seg/masks"
)

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.55
DEVICE = "cpu"


def load_model():
    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


def load_ground_truth(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")

    mask = mask.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.NEAREST,
    )

    mask = np.asarray(mask)

    return (mask > 0).astype(np.uint8)


def find_image(image_name: str) -> Path:
    matches = list(
        IMAGE_DIR.rglob(image_name)
    )

    if not matches:
        raise FileNotFoundError(
            f"Image not found: {image_name}"
        )

    return matches[0]


def find_mask(image_name: str) -> Path:
    mask_name = Path(image_name).stem + Path(
        image_name
    ).suffix

    matches = list(
        MASK_DIR.rglob(mask_name)
    )

    if not matches:
        raise FileNotFoundError(
            f"Mask not found: {mask_name}"
        )

    return matches[0]


def overlay_boundaries(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """
    Create an RGB overlay.

    Ground truth boundary:
        red channel

    Prediction boundary:
        blue channel

    Overlapping boundary:
        both red and blue
    """

    overlay = image.copy()

    gt_boundary = extract_boundary(
        ground_truth
    )

    pred_boundary = extract_boundary(
        prediction
    )

    # Ground-truth boundary
    overlay[gt_boundary] = (
        255,
        0,
        0,
    )

    # Prediction boundary
    overlay[pred_boundary] = (
        0,
        0,
        255,
    )

    # Overlapping boundary
    overlap = (
        gt_boundary
        & pred_boundary
    )

    overlay[overlap] = (
        255,
        0,
        255,
    )

    return overlay


def select_cases(df: pd.DataFrame):
    cases = []

    # ---------------------------------------------------------------
    # Case 1 — Worst Dice
    # ---------------------------------------------------------------

    worst_dice = df.nsmallest(
        1,
        "dice",
    ).iloc[0]

    cases.append(
        (
            "worst_dice",
            worst_dice,
        )
    )

    # ---------------------------------------------------------------
    # Case 2 — Worst ASSD
    # ---------------------------------------------------------------

    worst_assd = df.nlargest(
        1,
        "assd",
    ).iloc[0]

    cases.append(
        (
            "worst_assd",
            worst_assd,
        )
    )

    # ---------------------------------------------------------------
    # Case 3 — High Dice + Extreme Hausdorff
    # ---------------------------------------------------------------

    high_dice = df[
        df["dice"] >= 0.80
    ]

    high_dice_extreme_hd = (
        high_dice
        .nlargest(
            1,
            "hausdorff",
        )
        .iloc[0]
    )

    cases.append(
        (
            "high_dice_extreme_hausdorff",
            high_dice_extreme_hd,
        )
    )

    # ---------------------------------------------------------------
    # Case 4 — Very high Dice + extreme Hausdorff
    # ---------------------------------------------------------------

    very_high_dice = df[
        df["dice"] >= 0.90
    ]

    very_high_dice_extreme_hd = (
        very_high_dice
        .nlargest(
            1,
            "hausdorff",
        )
        .iloc[0]
    )

    cases.append(
        (
            "very_high_dice_extreme_hausdorff",
            very_high_dice_extreme_hd,
        )
    )

    # ---------------------------------------------------------------
    # Case 5 — Moderate Dice + very high ASSD
    # ---------------------------------------------------------------

    moderate_dice = df[
        (df["dice"] >= 0.30)
        & (df["dice"] <= 0.60)
    ]

    moderate_high_assd = (
        moderate_dice
        .nlargest(
            1,
            "assd",
        )
        .iloc[0]
    )

    cases.append(
        (
            "moderate_dice_high_assd",
            moderate_high_assd,
        )
    )

    # ---------------------------------------------------------------
    # Case 6 — Representative median Dice
    # ---------------------------------------------------------------

    median_dice = df["dice"].median()

    df_with_distance = df.copy()

    df_with_distance["median_distance"] = (
        df_with_distance["dice"]
        - median_dice
    ).abs()

    representative = (
        df_with_distance
        .nsmallest(
            1,
            "median_distance",
        )
        .iloc[0]
    )

    cases.append(
        (
            "representative_median_dice",
            representative,
        )
    )

    return cases


def visualize_case(
    case_name,
    row,
    segmenter,
):
    image_name = row["image_name"]

    image_path = find_image(
        image_name
    )

    mask_path = find_mask(
        image_name
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_array = np.asarray(
        image.resize(
            IMAGE_SIZE,
            resample=Image.Resampling.BILINEAR,
        )
    )

    ground_truth = load_ground_truth(
        mask_path
    )

    probability = segmenter.predict_proba(
        image
    )

    prediction = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    gt_boundary = extract_boundary(
        ground_truth
    )

    pred_boundary = extract_boundary(
        prediction
    )

    overlay = overlay_boundaries(
        image_array,
        ground_truth,
        prediction,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    axes[0, 0].imshow(
        image_array
    )
    axes[0, 0].set_title(
        "Original"
    )

    axes[0, 1].imshow(
        ground_truth,
        cmap="gray",
    )
    axes[0, 1].set_title(
        "Ground Truth"
    )

    axes[0, 2].imshow(
        prediction,
        cmap="gray",
    )
    axes[0, 2].set_title(
        "Prediction"
    )

    axes[1, 0].imshow(
        gt_boundary,
        cmap="gray",
    )
    axes[1, 0].set_title(
        "GT Boundary"
    )

    axes[1, 1].imshow(
        pred_boundary,
        cmap="gray",
    )
    axes[1, 1].set_title(
        "Prediction Boundary"
    )

    axes[1, 2].imshow(
        overlay
    )
    axes[1, 2].set_title(
        "Boundary Overlay"
    )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        (
            f"{case_name}\n"
            f"{image_name}\n"
            f"Dice={row['dice']:.4f} | "
            f"IoU={row['iou']:.4f} | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f}\n"
            f"ASSD={row['assd']:.4f} px | "
            f"Hausdorff={row['hausdorff']:.4f} px"
        ),
        fontsize=12,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{case_name}_{Path(image_name).stem}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def main():
    print("=" * 80)
    print("Day 42 — Boundary Failure Visualization")
    print("=" * 80)
    print()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing CSV: {CSV_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

    required_columns = [
        "image_name",
        "dice",
        "iou",
        "precision",
        "recall",
        "assd",
        "hausdorff",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = select_cases(
        df
    )

    print(
        f"Selected {len(cases)} cases:"
    )

    for case_name, row in cases:
        print(
            f"\n{case_name}"
        )
        print(
            f"  image    : {row['image_name']}"
        )
        print(
            f"  dice     : {row['dice']:.4f}"
        )
        print(
            f"  assd     : {row['assd']:.4f}"
        )
        print(
            f"  hausdorff: {row['hausdorff']:.4f}"
        )

    print()

    model = load_model()

    segmenter = Segmenter(
        model=model,
        device=DEVICE,
        threshold=THRESHOLD,
        image_size=IMAGE_SIZE,
    )

    for case_name, row in cases:
        visualize_case(
            case_name,
            row,
            segmenter,
        )

    print()
    print("=" * 80)
    print("Boundary visualization complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()