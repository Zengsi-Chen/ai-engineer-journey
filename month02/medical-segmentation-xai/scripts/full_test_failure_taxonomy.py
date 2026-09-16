from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from medseg.data.preprocessing import image_to_tensor, resize_image
from medseg.models.factory import create_model
from medseg.xai.saliency import compute_input_saliency


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "day37"
    / "baseline"
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

SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "split_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "day38_full_test_failure_taxonomy.csv"
)

IMAGE_SIZE = 256
THRESHOLD = 0.55

REASONABLE_DICE_THRESHOLD = 0.50
OVER_RATIO_THRESHOLD = 1.5
UNDER_RATIO_THRESHOLD = 0.5


def load_model():
    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_image(image_id):
    image_path = IMAGE_DIR / f"{image_id}.jpg"

    image = Image.open(
        image_path
    ).convert("RGB")

    resized = resize_image(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    tensor = image_to_tensor(
        resized
    ).unsqueeze(0)

    return tensor


def load_mask(image_id):
    mask_path = MASK_DIR / f"{image_id}.jpg"

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.NEAREST,
    )

    array = np.array(
        mask,
        copy=True,
    )

    return array > 0


def calculate_dice(gt_mask, prediction):
    gt_pixels = int(gt_mask.sum())
    pred_pixels = int(prediction.sum())

    intersection = np.logical_and(
        gt_mask,
        prediction,
    ).sum()

    denominator = gt_pixels + pred_pixels

    if denominator == 0:
        return 0.0

    return float(
        2.0 * intersection / denominator
    )


def classify_failure(
    dice,
    gt_pixels,
    pred_pixels,
):
    if dice >= REASONABLE_DICE_THRESHOLD:
        return "reasonable"

    if gt_pixels == 0:
        return "poor_localization"

    area_ratio = (
        pred_pixels / gt_pixels
        if gt_pixels > 0
        else 0.0
    )

    if area_ratio >= OVER_RATIO_THRESHOLD:
        return "over_segmentation"

    if area_ratio <= UNDER_RATIO_THRESHOLD:
        return "under_segmentation"

    return "poor_localization"


def analyze_case(model, image_id):
    image_tensor = load_image(
        image_id
    )

    gt_mask = load_mask(
        image_id
    )

    with torch.no_grad():
        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

        predicted_mask = (
            probabilities >= THRESHOLD
        )

    probability_map = (
        probabilities
        .squeeze()
        .cpu()
        .numpy()
    )

    prediction = (
        predicted_mask
        .squeeze()
        .cpu()
        .numpy()
    )

    saliency = compute_input_saliency(
        model=model,
        image=image_tensor,
        threshold=THRESHOLD,
        target_mode="soft_foreground",
    )

    saliency = (
        saliency
        .cpu()
        .numpy()
    )

    gt_pixels = int(
        gt_mask.sum()
    )

    pred_pixels = int(
        prediction.sum()
    )

    dice = calculate_dice(
        gt_mask=gt_mask,
        prediction=prediction,
    )

    iou_denominator = (
        gt_pixels
        + pred_pixels
        - np.logical_and(
            gt_mask,
            prediction,
        ).sum()
    )

    if iou_denominator > 0:
        iou = float(
            np.logical_and(
                gt_mask,
                prediction,
            ).sum()
            / iou_denominator
        )
    else:
        iou = 0.0

    if gt_pixels > 0:
        area_ratio = (
            pred_pixels / gt_pixels
        )
    else:
        area_ratio = 0.0

    saliency_inside = saliency[
        gt_mask
    ]

    saliency_outside = saliency[
        ~gt_mask
    ]

    inside_mean = float(
        saliency_inside.mean()
    )

    outside_mean = float(
        saliency_outside.mean()
    )

    if outside_mean > 0:
        concentration = (
            inside_mean
            / outside_mean
        )
    else:
        concentration = float("inf")

    failure_type = classify_failure(
        dice=dice,
        gt_pixels=gt_pixels,
        pred_pixels=pred_pixels,
    )

    return {
        "image_name": f"{image_id}.jpg",
        "dice": dice,
        "iou": iou,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "area_ratio": area_ratio,
        "max_probability": float(
            probability_map.max()
        ),
        "mean_probability": float(
            probability_map.mean()
        ),
        "saliency_inside_gt": inside_mean,
        "saliency_outside_gt": outside_mean,
        "saliency_concentration": concentration,
        "failure_type": failure_type,
    }


def main():
    print("=" * 80)
    print("Day 38 — Full Test-Set Failure Taxonomy")
    print("=" * 80)

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    manifest = pd.read_csv(
        SPLIT_MANIFEST
    )

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    print(
        f"Test samples: {len(test_df)}"
    )

    model = load_model()

    results = []

    for index, row in enumerate(
        test_df.itertuples(index=False),
        start=1,
    ):
        image_id = row.sample_id

        print(
            f"[{index}/{len(test_df)}] "
            f"{image_id}"
        )

        result = analyze_case(
            model=model,
            image_id=image_id,
        )

        results.append(
            result
        )

    df = pd.DataFrame(
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("Failure Type Summary")
    print("=" * 80)

    summary = (
        df.groupby("failure_type")
        .agg(
            count=("image_name", "count"),
            percentage=("image_name", lambda x: 100.0 * len(x) / len(df)),
            mean_dice=("dice", "mean"),
            mean_iou=("iou", "mean"),
            mean_area_ratio=("area_ratio", "mean"),
            mean_max_probability=("max_probability", "mean"),
            mean_saliency_concentration=(
                "saliency_concentration",
                "mean",
            ),
        )
        .sort_values(
            "count",
            ascending=False,
        )
    )

    print(
        summary.to_string()
    )

    print("\n" + "=" * 80)
    print("Dice Summary")
    print("=" * 80)

    print(
        df["dice"].describe()
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()