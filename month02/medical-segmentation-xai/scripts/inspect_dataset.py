from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medseg.data.validation import find_files


IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "masks"


def main():
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

    foreground_ratios = []
    empty_masks = []
    size_mismatches = []

    for image_path in images:
        mask_path = mask_map.get(image_path.stem)

        if mask_path is None:
            continue

        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask:
            mask_array = np.array(mask)

        mask_size = (
            mask_array.shape[1],
            mask_array.shape[0],
        )

        if image_size != mask_size:
            size_mismatches.append(
                image_path.name
            )

        foreground_pixels = np.count_nonzero(mask_array)
        total_pixels = mask_array.size

        ratio = (
            foreground_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        foreground_ratios.append(ratio)

        if foreground_pixels == 0:
            empty_masks.append(mask_path.name)

    print("=" * 60)
    print("Kvasir-SEG Dataset Audit")
    print("=" * 60)

    print(f"Images:           {len(images)}")
    print(f"Masks:            {len(masks)}")
    print(f"Matched pairs:    {len(foreground_ratios)}")
    print(f"Empty masks:      {len(empty_masks)}")
    print(f"Size mismatches:  {len(size_mismatches)}")

    if foreground_ratios:
        print()
        print("Foreground Ratio")
        print("-" * 60)
        print(f"Min:     {min(foreground_ratios):.4f}")
        print(f"Mean:    {np.mean(foreground_ratios):.4f}")
        print(f"Median:  {np.median(foreground_ratios):.4f}")
        print(f"Max:     {max(foreground_ratios):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()