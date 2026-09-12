from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch

from medseg.models.factory import create_model
from medseg.inference.segmenter import Segmenter


CSV_PATH = Path(
    "artifacts/reports/day36_error_analysis.csv"
)

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/best_model.pt"
)

OUTPUT_DIR = Path(
    "artifacts/figures/day36_error_cases"
)

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.55
NUM_CASES = 5

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


def load_ground_truth(mask_path):
    mask = Image.open(mask_path).convert("L")

    mask = mask.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.NEAREST,
    )

    mask = np.asarray(mask)

    return (mask > 0).astype(np.uint8)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing CSV: {CSV_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "image_name",
        "dice",
        "iou",
        "precision",
        "recall",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    worst_cases = (
        df.sort_values("dice", ascending=True)
        .head(NUM_CASES)
    )

    print("=" * 80)
    print("Day 36 — Visual Error Analysis")
    print("=" * 80)

    print(
        f"\nAnalyzing worst {NUM_CASES} cases by Dice"
    )

    print(
        worst_cases[
            [
                "image_name",
                "dice",
                "iou",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model()

    segmenter = Segmenter(
        model=model,
        device=DEVICE,
        threshold=THRESHOLD,
        image_size=IMAGE_SIZE,
    )

    for _, row in worst_cases.iterrows():

        image_path = Path(row["image_name"])

        # Search for the actual image in the Kvasir dataset.
        matches = list(
            Path("data/raw/kvasir-seg/images").rglob(
                image_path.name
            )
        )

        if not matches:
            print(
                f"WARNING: image not found: "
                f"{image_path.name}"
            )
            continue

        image_path = matches[0]

        mask_matches = list(
            Path("data/raw/kvasir-seg/masks").rglob(
                image_path.stem + image_path.suffix
            )
        )

        if not mask_matches:
            print(
                f"WARNING: mask not found: "
                f"{image_path.name}"
            )
            continue

        mask_path = mask_matches[0]

        image = Image.open(
            image_path
        ).convert("RGB")

        ground_truth = load_ground_truth(
            mask_path
        )

        probability = segmenter.predict_proba(
            image
        )

        prediction = (
            probability >= THRESHOLD
        ).astype(np.uint8)

        # ============================================================
        # Complete Miss Quantitative Analysis
        # ============================================================

        gt_pixels = int(ground_truth.sum())
        pred_pixels = int(prediction.sum())

        gt_ratio = gt_pixels / ground_truth.size
        pred_ratio = pred_pixels / prediction.size

        prob_min = float(probability.min())
        prob_max = float(probability.max())
        prob_mean = float(probability.mean())

        # Probability statistics inside GT region
        if gt_pixels > 0:
            gt_probabilities = probability[ground_truth > 0]

            gt_prob_mean = float(gt_probabilities.mean())
            gt_prob_max = float(gt_probabilities.max())
            gt_prob_min = float(gt_probabilities.min())
        else:
            gt_prob_mean = 0.0
            gt_prob_max = 0.0
            gt_prob_min = 0.0

        print("\n" + "-" * 80)
        print(f"Complete Miss Analysis: {image_path.name}")
        print("-" * 80)

        print(f"GT foreground pixels       : {gt_pixels}")
        print(f"Prediction foreground      : {pred_pixels}")

        print(f"GT foreground ratio        : {gt_ratio:.6f}")
        print(f"Prediction foreground ratio: {pred_ratio:.6f}")

        print(f"Probability min            : {prob_min:.6f}")
        print(f"Probability max            : {prob_max:.6f}")
        print(f"Probability mean           : {prob_mean:.6f}")

        print(f"Probability in GT region:")
        print(f"  min                      : {gt_prob_min:.6f}")
        print(f"  mean                     : {gt_prob_mean:.6f}")
        print(f"  max                      : {gt_prob_max:.6f}")

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 10),
        )

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original")

        axes[0, 1].imshow(
            ground_truth,
            cmap="gray",
        )
        axes[0, 1].set_title("Ground Truth")

        axes[1, 0].imshow(
            probability,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, 0].set_title(
            f"Probability Map (T={THRESHOLD})"
        )

        axes[1, 1].imshow(
            prediction,
            cmap="gray",
        )
        axes[1, 1].set_title(
            "Prediction"
        )

        for ax in axes.ravel():
            ax.axis("off")

        fig.suptitle(
            (
                f"{image_path.name}\n"
                f"Dice={row['dice']:.4f} | "
                f"IoU={row['iou']:.4f} | "
                f"Precision={row['precision']:.4f} | "
                f"Recall={row['recall']:.4f}"
            ),
            fontsize=12,
        )

        fig.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_error_analysis.png"
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

    print("\n" + "=" * 80)
    print("Visual error analysis complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()