from pathlib import Path

import numpy as np
from PIL import Image

from medseg.data.validation import (
    find_files,
    validate_image_mask_pair,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "masks"


def test_all_image_mask_sizes_match():
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

    mismatches = []

    for image in images:
        mask = mask_map.get(image.stem)

        if mask is None:
            continue

        result = validate_image_mask_pair(
            image,
            mask,
        )

        if not result["size_match"]:
            mismatches.append(
                {
                    "image": str(image),
                    "mask": str(mask),
                    "image_size": result["image_size"],
                    "mask_size": result["mask_size"],
                }
            )

    assert mismatches == []



def test_no_empty_masks():
    masks = find_files(
        MASK_DIR,
        {".jpg", ".jpeg", ".png"},
    )

    empty_masks = []

    for mask in masks:
        with Image.open(mask) as image:
            import numpy as np

            array = np.array(image)

        if np.count_nonzero(array) == 0:
            empty_masks.append(str(mask))

    assert empty_masks == []