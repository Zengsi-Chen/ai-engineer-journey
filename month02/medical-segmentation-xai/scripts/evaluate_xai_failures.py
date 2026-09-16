from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from medseg.data.preprocessing import image_to_tensor, resize_image
from medseg.models.factory import create_model
from medseg.xai.saliency import compute_input_saliency


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATHS = {
    "day35": PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt",
    "day37_baseline": PROJECT_ROOT / "artifacts" / "day37" / "baseline" / "best_model.pt",
}

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "masks"

OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "day38_xai_checkpoint_comparison.csv"

IMAGE_SIZE = 256
THRESHOLD = 0.55

FAILURE_CASES = [
    "cju76o55nymqd0871h31sph9w",
    "cju887ftknop008177nnjt46y",
    "cju35k2fr3vc50988c85qkrwg",
    "cju1egh885m1l0855ci1lt37c",
    "cju2zpw4q9vzr0801p0lysjdl",
    "cju2saez63gxl08559ucjq3kt",
]


def load_model(checkpoint_path):
    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    checkpoint = torch.load(
        checkpoint_path,
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

    return resized, tensor


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


def analyze_case(model, image_id):
    image, image_tensor = load_image(
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

    dice = (
        2.0
        * np.logical_and(
            gt_mask,
            prediction,
        ).sum()
        / (
            gt_pixels
            + pred_pixels
        )
        if (gt_pixels + pred_pixels) > 0
        else 0.0
    )

    saliency_inside_gt = saliency[
        gt_mask
    ]

    saliency_outside_gt = saliency[
        ~gt_mask
    ]

    inside_mean = float(
        saliency_inside_gt.mean()
    )

    outside_mean = float(
        saliency_outside_gt.mean()
    )

    if outside_mean > 0:
        concentration = (
            inside_mean
            / outside_mean
        )
    else:
        concentration = float("inf")

    return {
        "image_name": f"{image_id}.jpg",
        "dice": dice,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "max_probability": float(
            probability_map.max()
        ),
        "mean_probability": float(
            probability_map.mean()
        ),
        "saliency_inside_gt": inside_mean,
        "saliency_outside_gt": outside_mean,
        "saliency_concentration": concentration,
    }


def main():
    print("=" * 80)
    print("Day 38 — Batch XAI Checkpoint Comparison")
    print("=" * 80)

    print(
        f"Cases: {len(FAILURE_CASES)}"
    )

    all_results = []

    for checkpoint_name, checkpoint_path in CHECKPOINT_PATHS.items():
        print()
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_name}")
        print(f"Path: {checkpoint_path}")
        print("=" * 80)

        model = load_model(
            checkpoint_path
        )

        for index, image_id in enumerate(
            FAILURE_CASES,
            start=1,
        ):
            print(
                f"\n[{index}/{len(FAILURE_CASES)}] "
                f"{image_id}"
            )

            result = analyze_case(
                model=model,
                image_id=image_id,
            )

            result["checkpoint"] = checkpoint_name

            all_results.append(
                result
            )

            print(
                f"Dice: {result['dice']:.4f}"
            )

            print(
                "Max probability: "
                f"{result['max_probability']:.6f}"
            )

            print(
                "Saliency concentration: "
                f"{result['saliency_concentration']:.3f}x"
            )

    df = pd.DataFrame(
        all_results
    )

    columns = [
        "checkpoint",
        "image_name",
        "dice",
        "gt_pixels",
        "pred_pixels",
        "max_probability",
        "mean_probability",
        "saliency_inside_gt",
        "saliency_outside_gt",
        "saliency_concentration",
    ]

    df = df[columns]

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