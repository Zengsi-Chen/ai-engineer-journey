from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from medseg.data.preprocessing import image_to_tensor, resize_image
from medseg.models.factory import create_model


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "day38_threshold_sensitivity.csv"
)

IMAGE_SIZE = 256

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
    0.70,
]

FAILURE_CASES = [
    "cju76o55nymqd0871h31sph9w",
    "cju887ftknop008177nnjt46y",
    "cju35k2fr3vc50988c85qkrwg",
    "cju1egh885m1l0855ci1lt37c",
    "cju2zpw4q9vzr0801p0lysjdl",
    "cju2saez63gxl08559ucjq3kt",
]


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


def analyze_case(model, image_id, probabilities):
    gt_mask = load_mask(image_id)

    gt_pixels = int(
        gt_mask.sum()
    )

    max_probability = float(
        probabilities.max()
    )

    results = []

    for threshold in THRESHOLDS:
        prediction = (
            probabilities >= threshold
        )

        pred_pixels = int(
            prediction.sum()
        )

        dice = calculate_dice(
            gt_mask=gt_mask,
            prediction=prediction,
        )

        results.append(
            {
                "image_name": f"{image_id}.jpg",
                "threshold": threshold,
                "dice": dice,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "max_probability": max_probability,
                "empty_prediction": pred_pixels == 0,
            }
        )

    return results


def main():
    print("=" * 80)
    print("Day 38 — Threshold Sensitivity Analysis")
    print("=" * 80)

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Thresholds: {THRESHOLDS}"
    )

    model = load_model()

    all_results = []

    for index, image_id in enumerate(
        FAILURE_CASES,
        start=1,
    ):
        print(
            f"\n[{index}/{len(FAILURE_CASES)}] "
            f"{image_id}"
        )

        image_tensor = load_image(
            image_id
        )

        with torch.no_grad():
            logits = model(
                image_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

        probability_map = (
            probabilities
            .squeeze()
            .cpu()
            .numpy()
        )

        case_results = analyze_case(
            model=model,
            image_id=image_id,
            probabilities=probability_map,
        )

        all_results.extend(
            case_results
        )

        for result in case_results:
            print(
                f"threshold={result['threshold']:.2f} "
                f"Dice={result['dice']:.4f} "
                f"pred_pixels={result['pred_pixels']}"
            )

    df = pd.DataFrame(
        all_results
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
    print("Summary")
    print("=" * 80)

    print(
        df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()