"""
Boundary-aware metrics for binary segmentation masks.

This module provides lightweight boundary analysis utilities based on
NumPy and SciPy. It intentionally avoids adding a new dependency such
as scikit-image.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _validate_masks(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert prediction and target masks."""

    if prediction.ndim != 2:
        raise ValueError(
            f"prediction must be 2D, got shape {prediction.shape}"
        )

    if target.ndim != 2:
        raise ValueError(
            f"target must be 2D, got shape {target.shape}"
        )

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape: "
            f"{prediction.shape} != {target.shape}"
        )

    prediction_binary = prediction > 0
    target_binary = target > 0

    return prediction_binary, target_binary


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract the one-pixel inner boundary of a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask. Non-zero values are treated as foreground.

    Returns
    -------
    np.ndarray
        Boolean boundary mask.
    """

    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2D, got shape {mask.shape}"
        )

    binary_mask = mask > 0

    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    structure = np.ones((3, 3), dtype=bool)
    eroded_mask = ndimage.binary_erosion(
        binary_mask,
        structure=structure,
        border_value=0,
    )

    return binary_mask & ~eroded_mask


def boundary_distances(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate nearest-boundary distances in both directions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Prediction-to-target distances and target-to-prediction
        distances.

    Raises
    ------
    ValueError
        If either mask has no foreground pixels.
    """

    prediction_binary, target_binary = _validate_masks(
        prediction,
        target,
    )

    prediction_boundary = extract_boundary(prediction_binary)
    target_boundary = extract_boundary(target_binary)

    if not np.any(prediction_boundary):
        raise ValueError("prediction has no boundary")

    if not np.any(target_boundary):
        raise ValueError("target has no boundary")

    target_distance_map = ndimage.distance_transform_edt(
        ~target_boundary
    )

    prediction_distance_map = ndimage.distance_transform_edt(
        ~prediction_boundary
    )

    prediction_to_target = target_distance_map[prediction_boundary]
    target_to_prediction = prediction_distance_map[target_boundary]

    return prediction_to_target, target_to_prediction


def average_symmetric_surface_distance(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Calculate Average Symmetric Surface Distance (ASSD).

    Lower values indicate closer boundary agreement.
    """

    prediction_to_target, target_to_prediction = boundary_distances(
        prediction,
        target,
    )

    all_distances = np.concatenate(
        [prediction_to_target, target_to_prediction]
    )

    return float(np.mean(all_distances))


def hausdorff_distance(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Calculate symmetric Hausdorff distance between boundaries.

    Lower values indicate closer boundary agreement.
    """

    prediction_to_target, target_to_prediction = boundary_distances(
        prediction,
        target,
    )

    return float(
        max(
            np.max(prediction_to_target),
            np.max(target_to_prediction),
        )
    )