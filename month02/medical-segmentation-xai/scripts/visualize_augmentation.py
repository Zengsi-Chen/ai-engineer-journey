from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from medseg.data.augmentation import SegmentationAugmentation
from medseg.data.preprocessing import resize_image, resize_mask


MANIFEST_PATH = Path("artifacts/splits/split_manifest.csv")
OUTPUT_PATH = Path("artifacts/reports/day37_augmentation_visual_check.png")

IMAGE_SIZE = 256
SAMPLE_INDEX = 0


def load_sample(manifest_path: Path, split: str, index: int):
    manifest = pd.read_csv(manifest_path)

    samples = manifest[manifest["split"] == split].reset_index(drop=True)

    row = samples.iloc[index]

    image = Image.open(row["image_path"]).convert("RGB")
    mask = Image.open(row["mask_path"]).convert("L")

    size = (IMAGE_SIZE, IMAGE_SIZE)

    image = resize_image(image, size)
    mask = resize_mask(mask, size)

    return image, mask, row["sample_id"]


def mask_to_numpy(mask: Image.Image):
    array = np.asarray(mask)
    return array > 0


def create_overlay(image: Image.Image, mask: Image.Image):
    image_array = np.asarray(image).copy()
    mask_array = mask_to_numpy(mask)

    overlay = image_array.copy()

    # Highlight mask region by increasing red channel
    overlay[mask_array, 0] = 255

    return overlay


def main():
    print("=" * 80)
    print("Day 37 — Real Kvasir-SEG Augmentation Visual Check")
    print("=" * 80)

    image, mask, sample_id = load_sample(
        MANIFEST_PATH,
        split="train",
        index=SAMPLE_INDEX,
    )

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        rotation_prob=0.5,
        rotation_degrees=15.0,
    )

    augmented_image, augmented_mask = augmentation(
        image.copy(),
        mask.copy(),
    )

    original_overlay = create_overlay(image, mask)
    augmented_overlay = create_overlay(
        augmented_image,
        augmented_mask,
    )

    print(f"Sample ID: {sample_id}")
    print(f"Original image size: {image.size}")
    print(f"Original mask size: {mask.size}")
    print(f"Augmented image size: {augmented_image.size}")
    print(f"Augmented mask size: {augmented_mask.size}")

    original_mask_pixels = mask_to_numpy(mask).sum()
    augmented_mask_pixels = mask_to_numpy(augmented_mask).sum()

    print(f"Original mask pixels:   {original_mask_pixels}")
    print(f"Augmented mask pixels:  {augmented_mask_pixels}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Original Mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(original_overlay)
    axes[0, 2].set_title("Original Overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(augmented_image)
    axes[1, 0].set_title("Augmented Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(augmented_mask, cmap="gray")
    axes[1, 1].set_title("Augmented Mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(augmented_overlay)
    axes[1, 2].set_title("Augmented Overlay")
    axes[1, 2].axis("off")

    fig.suptitle(
        f"Kvasir-SEG Augmentation Spatial Alignment — {sample_id}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print()
    print(f"Saved visualization to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()