import numpy as np
import pytest

from medseg.evaluation.boundary_metrics import (
    average_symmetric_surface_distance,
    boundary_distances,
    extract_boundary,
    hausdorff_distance,
)


def test_extract_boundary_empty_mask():
    mask = np.zeros((8, 8), dtype=np.uint8)

    boundary = extract_boundary(mask)

    assert boundary.shape == mask.shape
    assert boundary.dtype == bool
    assert not np.any(boundary)


def test_extract_boundary_single_pixel():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[4, 4] = 1

    boundary = extract_boundary(mask)

    assert boundary.sum() == 1
    assert boundary[4, 4]


def test_extract_boundary_filled_square():
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 1

    boundary = extract_boundary(mask)

    # A 5x5 square eroded once becomes a 3x3 square.
    # Therefore, the inner boundary contains 25 - 9 = 16 pixels.
    assert boundary.sum() == 16


def test_identical_masks_have_zero_boundary_distance():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1

    prediction_to_target, target_to_prediction = boundary_distances(
        mask,
        mask,
    )

    assert np.allclose(prediction_to_target, 0.0)
    assert np.allclose(target_to_prediction, 0.0)


def test_identical_masks_have_zero_assd_and_hausdorff():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1

    assert average_symmetric_surface_distance(mask, mask) == 0.0
    assert hausdorff_distance(mask, mask) == 0.0


def test_shifted_masks_have_positive_boundary_distance():
    target = np.zeros((20, 20), dtype=np.uint8)
    prediction = np.zeros((20, 20), dtype=np.uint8)

    target[5:11, 5:11] = 1
    prediction[5:11, 6:12] = 1

    assd = average_symmetric_surface_distance(
        prediction,
        target,
    )
    hausdorff = hausdorff_distance(
        prediction,
        target,
    )

    assert assd > 0.0
    assert hausdorff > 0.0


def test_boundary_metrics_reject_shape_mismatch():
    prediction = np.zeros((8, 8), dtype=np.uint8)
    target = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        boundary_distances(prediction, target)


def test_boundary_metrics_reject_empty_prediction():
    prediction = np.zeros((8, 8), dtype=np.uint8)
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2:6, 2:6] = 1

    with pytest.raises(ValueError, match="prediction has no boundary"):
        boundary_distances(prediction, target)


def test_boundary_metrics_reject_empty_target():
    prediction = np.zeros((8, 8), dtype=np.uint8)
    target = np.zeros((8, 8), dtype=np.uint8)
    prediction[2:6, 2:6] = 1

    with pytest.raises(ValueError, match="target has no boundary"):
        boundary_distances(prediction, target)