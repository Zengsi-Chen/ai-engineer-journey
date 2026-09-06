from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medseg.data.validation import find_files


IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "masks"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "figures"


def create_overlay(
    image_array: np.ndarray,
    mask_array: np.ndarray,
) -> np.ndarray:
    """
    Create an RGB overlay from an image and binary segmentation mask.
    """

    # Ensure RGB image.
    if image_array.ndim == 2:
        image_array = np.stack(
            [image_array] * 3,
            axis=-1,
        )

    # Convert mask to boolean foreground.
    foreground = mask_array > 0

    # Create a red mask with the same shape as the RGB image.
    red_mask = np.zeros_like(
        image_array,
        dtype=np.uint8,
    )

    red_mask[..., 0] = 255

    # Convert image to float for blending.
    image_float = image_array.astype(
        np.float32
    )

    red_float = red_mask.astype(
        np.float32
    )

    # Expand 2D foreground mask into 3 channels.
    foreground_3d = np.repeat(
        foreground[..., np.newaxis],
        3,
        axis=2,
    )

    # Blend original image with red mask.
    blended = (
        0.5 * image_float
        + 0.5 * red_float
    )

    # Only apply blending to foreground.
    overlay = np.where(
        foreground_3d,
        blended,
        image_float,
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = find_files(
        IMAGE_DIR,
        {".jpg", ".jpeg", ".png"},
    )

    masks = find_files(
        MASK_DIR,
        {".jpg", ".jpeg", ".png"},
    )

    mask_map = {
        mask.stem: mask
        for mask in masks
    }

    valid_pairs = [
        (image, mask_map[image.stem])
        for image in images
        if image.stem in mask_map
    ]

    if not valid_pairs:
        raise RuntimeError(
            "No valid image-mask pairs found."
        )

    random.seed(42)

    samples = random.sample(
        valid_pairs,
        min(6, len(valid_pairs)),
    )

    for index, (image_path, mask_path) in enumerate(
        samples,
        start=1,
    ):
        with Image.open(image_path) as image:
            image_array = np.array(
                image.convert("RGB")
            )

        with Image.open(mask_path) as mask:
            mask_array = np.array(mask.convert("L"))

        overlay = create_overlay(
            image_array,
            mask_array,
        )

        print(
            f"Sample {index}: "
            f"image={image_array.shape}, "
            f"mask={mask_array.shape}, "
            f"foreground_ratio="
            f"{np.mean(mask_array > 0):.4f}"
        )

        fig = plt.figure(figsize=(12, 4))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(image_array)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(mask_array, cmap="gray")
        ax2.set_title("Segmentation Mask")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(overlay)
        ax3.set_title("Image + Mask Overlay")
        ax3.axis("off")

        fig.suptitle(image_path.name)

        output_path = (
            OUTPUT_DIR
            / f"sample_{index}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()