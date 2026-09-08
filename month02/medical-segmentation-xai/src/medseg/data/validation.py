from pathlib import Path
from PIL import Image

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_files(directory: Path, extensions: set[str]) -> list[Path]:
    """Find files recursively with supported extensions."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def validate_image_file(path: Path) -> bool:
    """Check whether an image can be opened successfully."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def validate_dataset_structure(
    image_dir: Path,
    mask_dir: Path,
) -> dict:
    """
    Validate basic image/mask integrity.

    Returns a dictionary containing:
    - image_count
    - mask_count
    - broken_images
    - broken_masks
    - missing_masks
    - orphan_masks
    - matched_pairs
    """

    images = find_files(image_dir, IMAGE_EXTENSIONS)
    masks = find_files(mask_dir, MASK_EXTENSIONS)

    image_map = {path.stem: path for path in images}
    mask_map = {path.stem: path for path in masks}

    image_names = set(image_map)
    mask_names = set(mask_map)

    missing_masks = sorted(image_names - mask_names)
    orphan_masks = sorted(mask_names - image_names)

    broken_images = [
        str(path)
        for path in images
        if not validate_image_file(path)
    ]

    broken_masks = [
        str(path)
        for path in masks
        if not validate_image_file(path)
    ]

    matched_pairs = sorted(image_names & mask_names)

    return {
        "image_count": len(images),
        "mask_count": len(masks),
        "broken_images": broken_images,
        "broken_masks": broken_masks,
        "missing_masks": missing_masks,
        "orphan_masks": orphan_masks,
        "matched_pairs": matched_pairs,
    }


def inspect_mask(mask_path: Path) -> dict:
    """
    Inspect a segmentation mask.

    Returns:
    - width
    - height
    - unique_values
    - foreground_pixels
    - total_pixels
    - foreground_ratio
    - is_empty
    """

    with Image.open(mask_path) as image:
        mask = np.array(image)

    unique_values = np.unique(mask)

    # Any non-zero pixel is treated as foreground.
    foreground_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)

    foreground_ratio = (
        foreground_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return {
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "unique_values": unique_values.tolist(),
        "foreground_pixels": foreground_pixels,
        "total_pixels": total_pixels,
        "foreground_ratio": foreground_ratio,
        "is_empty": foreground_pixels == 0,
    }


def validate_image_mask_pair(
    image_path: Path,
    mask_path: Path,
) -> dict:
    """
    Validate one image/mask pair.
    """

    with Image.open(image_path) as image:
        image_size = image.size

    mask_info = inspect_mask(mask_path)

    mask_size = (
        mask_info["width"],
        mask_info["height"],
    )

    return {
        "image_size": image_size,
        "mask_size": mask_size,
        "size_match": image_size == mask_size,
        "mask_info": mask_info,
    }

