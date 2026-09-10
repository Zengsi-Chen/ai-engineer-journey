from pathlib import Path

import numpy as np
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


def main():

    print("=" * 70)
    print("Day 35 Threshold Sweep")
    print("=" * 70)

    model = load_model(
        CHECKPOINT_PATH
    )

    # threshold is temporary here;
    # we use predict_proba() and apply
    # different thresholds ourselves.
    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=0.5,
        image_size=IMAGE_SIZE,
    )

    images = sorted(
        IMAGE_DIR.glob("*")
    )

    if not images:
        raise FileNotFoundError(
            f"No images found in {IMAGE_DIR}"
        )

    image_path = images[0]

    mask_path = (
        MASK_DIR
        / image_path.name
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    target = load_ground_truth(
        mask_path
    )

    # IMPORTANT:
    # Run the model only ONCE.
    probabilities = (
        segmenter.predict_proba(image)
    )

    print()
    print(f"Image: {image_path.name}")
    print()

    print(
        f"{'Threshold':>10} "
        f"{'Dice':>10} "
        f"{'IoU':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'FG Ratio':>12}"
    )

    print("-" * 70)

    results = []

    for threshold in THRESHOLDS:

        prediction = (
            probabilities >= threshold
        ).astype(np.uint8)

        metrics = calculate_metrics(
            prediction,
            target,
        )

        foreground_ratio = (
            prediction.mean()
        )

        result = {
            "threshold": threshold,
            **metrics,
            "foreground_ratio": float(
                foreground_ratio
            ),
        }

        results.append(result)

        print(
            f"{threshold:10.2f} "
            f"{metrics['dice']:10.4f} "
            f"{metrics['iou']:10.4f} "
            f"{metrics['precision']:10.4f} "
            f"{metrics['recall']:10.4f} "
            f"{foreground_ratio:11.4%}"
        )

    best = max(
        results,
        key=lambda x: x["dice"],
    )

    print()
    print("=" * 70)
    print("Best Threshold")
    print("=" * 70)

    print(
        f"Threshold:  {best['threshold']:.2f}"
    )

    print(
        f"Dice:       {best['dice']:.4f}"
    )

    print(
        f"IoU:        {best['iou']:.4f}"
    )

    print(
        f"Precision:  {best['precision']:.4f}"
    )

    print(
        f"Recall:      {best['recall']:.4f}"
    )

    print(
        f"FG Ratio:   "
        f"{best['foreground_ratio']:.4%}"
    )


if __name__ == "__main__":
    main()