import numpy as np
from PIL import Image

from medseg.data.augmentation import SegmentationAugmentation


def create_test_image_and_mask():
    image_array = np.zeros((32, 32, 3), dtype=np.uint8)
    mask_array = np.zeros((32, 32), dtype=np.uint8)

    image_array[8:20, 5:15] = 255
    mask_array[8:20, 5:15] = 255

    image = Image.fromarray(image_array, mode="RGB")
    mask = Image.fromarray(mask_array, mode="L")

    return image, mask


def test_augmentation_is_deterministic_when_disabled():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_prob=0.0,
    )

    image_1, mask_1 = augmentation(image.copy(), mask.copy())
    image_2, mask_2 = augmentation(image.copy(), mask.copy())

    assert np.array_equal(
        np.asarray(image_1),
        np.asarray(image_2),
    )

    assert np.array_equal(
        np.asarray(mask_1),
        np.asarray(mask_2),
    )


def test_horizontal_flip_keeps_image_and_mask_aligned():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=1.0,
        vertical_flip_prob=0.0,
        rotation_prob=0.0,
    )

    augmented_image, augmented_mask = augmentation(
        image.copy(),
        mask.copy(),
    )

    image_array = np.asarray(augmented_image)
    mask_array = np.asarray(augmented_mask)

    image_foreground = image_array[:, :, 0] > 0
    mask_foreground = mask_array > 0

    assert np.array_equal(
        image_foreground,
        mask_foreground,
    )


def test_vertical_flip_keeps_image_and_mask_aligned():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=0.0,
        vertical_flip_prob=1.0,
        rotation_prob=0.0,
    )

    augmented_image, augmented_mask = augmentation(
        image.copy(),
        mask.copy(),
    )

    image_array = np.asarray(augmented_image)
    mask_array = np.asarray(augmented_mask)

    image_foreground = image_array[:, :, 0] > 0
    mask_foreground = mask_array > 0

    assert np.array_equal(
        image_foreground,
        mask_foreground,
    )


def test_rotation_keeps_image_and_mask_aligned():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_prob=1.0,
        rotation_degrees=15.0,
    )

    augmented_image, augmented_mask = augmentation(
        image.copy(),
        mask.copy(),
    )

    image_array = np.asarray(augmented_image)
    mask_array = np.asarray(augmented_mask)

    image_foreground = image_array[:, :, 0] > 0
    mask_foreground = mask_array > 0

    intersection = np.logical_and(
        image_foreground,
        mask_foreground,
    ).sum()

    mask_area = mask_foreground.sum()

    assert intersection / mask_area > 0.95