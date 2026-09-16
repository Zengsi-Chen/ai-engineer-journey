import numpy as np
import pytest

from medseg.postprocessing.connected_components import (
    remove_small_components,
)


def test_min_area_zero_keeps_mask():
    mask = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )

    result = remove_small_components(mask, min_area=0)

    np.testing.assert_array_equal(result, mask)


def test_removes_small_component():
    mask = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = remove_small_components(mask, min_area=2)

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_keeps_component_equal_to_min_area():
    mask = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )

    result = remove_small_components(mask, min_area=2)

    expected = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_diagonal_pixels_are_connected():
    mask = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = remove_small_components(mask, min_area=2)

    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected)


def test_empty_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)

    result = remove_small_components(mask, min_area=10)

    expected = np.zeros((4, 4), dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_rejects_non_2d_mask():
    mask = np.zeros((1, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        remove_small_components(mask, min_area=2)


def test_rejects_negative_min_area():
    mask = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        remove_small_components(mask, min_area=-1)