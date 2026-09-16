"""
Connected-component based post-processing for binary segmentation masks.
"""

import numpy as np
from scipy import ndimage


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """
    Remove connected components smaller than min_area.

    Parameters
    ----------
    mask : np.ndarray
        2D binary segmentation mask.
        Foreground should be represented by non-zero values.

    min_area : int
        Minimum component area in pixels.
        Components smaller than this value are removed.
        min_area=0 keeps the original mask unchanged.

    Returns
    -------
    np.ndarray
        Cleaned binary mask with values 0 or 1.

    Raises
    ------
    ValueError
        If the mask is not 2D or min_area is negative.
    """

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got shape {mask.shape}"
        )

    if min_area < 0:
        raise ValueError(
            f"min_area must be >= 0, got {min_area}"
        )

    binary_mask = mask > 0

    if min_area == 0:
        return binary_mask.astype(np.uint8)

    # 8-connectivity:
    # diagonal foreground pixels are considered connected.
    structure = np.ones((3, 3), dtype=np.uint8)

    labeled_mask, num_components = ndimage.label(
        binary_mask,
        structure=structure,
    )

    if num_components == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    component_sizes = np.bincount(
        labeled_mask.ravel()
    )

    keep_components = component_sizes >= min_area
    keep_components[0] = False  # background

    cleaned_mask = keep_components[labeled_mask]

    return cleaned_mask.astype(np.uint8)