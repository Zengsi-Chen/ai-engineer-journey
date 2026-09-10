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
THRESHOLD = 0.5


def load_model(
    checkpoint_path: Path,
) -> torch.nn.Module:

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

    return (array > 0).astype(np.uint8)


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:

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

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * true_positive
            / dice_denominator
        )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    if union == 0:
        iou = 1.0
    else:
        iou = (
            true_positive
            / union
        )

    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            true_positive
            / precision_denominator
        )

    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            true_positive
            / recall_denominator
        )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def main() -> None:

    print("=" * 60)
    print("Day 35 Single Image Evaluation")
    print("=" * 60)

    model = load_model(
        CHECKPOINT_PATH,
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=THRESHOLD,
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
        MASK_DIR / image_path.name
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {mask_path}"
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

    print()
    print(f"Image: {image_path.name}")
    print(f"Threshold: {THRESHOLD}")
    print()

    print(
        f"GT foreground ratio:   "
        f"{target.mean():.4%}"
    )

    print(
        f"Prediction foreground:  "
        f"{prediction.mean():.4%}"
    )

    print()

    print(
        f"Dice:       {metrics['dice']:.4f}"
    )

    print(
        f"IoU:        {metrics['iou']:.4f}"
    )

    print(
        f"Precision:  {metrics['precision']:.4f}"
    )

    print(
        f"Recall:     {metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()