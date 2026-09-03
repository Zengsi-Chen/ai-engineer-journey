import numpy as np
import pytest

from src.augmentation import TrainOnlyAugmenter


def create_test_images():
    return np.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ],
        [
            [
                [7, 8, 9],
                [10, 11, 12],
            ]
        ],
    ])


def test_train_images_are_augmented():
    images = create_test_images()

    augmenter = TrainOnlyAugmenter(
        seed=42,
        probability=1.0,
    )

    augmented = augmenter.transform(
        images,
        split="train",
    )

    expected = np.array([
        [
            [
                [3, 2, 1],
                [6, 5, 4],
            ]
        ],
        [
            [
                [9, 8, 7],
                [12, 11, 10],
            ]
        ],
    ])

    np.testing.assert_array_equal(
        augmented,
        expected,
    )


def test_validation_images_are_not_augmented():
    images = create_test_images()

    augmenter = TrainOnlyAugmenter(
        probability=1.0,
    )

    result = augmenter.transform(
        images,
        split="validation",
    )

    np.testing.assert_array_equal(
        result,
        images,
    )


def test_test_images_are_not_augmented():
    images = create_test_images()

    augmenter = TrainOnlyAugmenter(
        probability=1.0,
    )

    result = augmenter.transform(
        images,
        split="test",
    )

    np.testing.assert_array_equal(
        result,
        images,
    )


def test_unknown_split_raises_error():
    images = create_test_images()

    augmenter = TrainOnlyAugmenter()

    with pytest.raises(ValueError):
        augmenter.transform(
            images,
            split="all_data",
        )


def test_validation_is_deterministic():
    images = np.arange(
        24,
        dtype=np.float32,
    ).reshape(2, 1, 3, 4)

    augmenter = TrainOnlyAugmenter(
        seed=42,
        probability=1.0,
    )

    result_1 = augmenter.transform(
        images,
        split="validation",
    )

    result_2 = augmenter.transform(
        images,
        split="validation",
    )

    assert np.array_equal(
        result_1,
        result_2,
    )

    assert np.array_equal(
        result_1,
        images,
    )
    