from PIL import Image
import numpy as np

from medseg.data.augmentation import SegmentationAugmentation


def create_test_image_and_mask():
    """
    Create a tiny synthetic image and mask with an asymmetric pattern.

    The asymmetric pattern makes it possible to detect whether
    image and mask receive the same geometric transformation.
    """

    image_array = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_array = np.zeros((8, 8), dtype=np.uint8)

    # Create a small asymmetric foreground region.
    image_array[2:5, 1:4] = 255
    mask_array[2:5, 1:4] = 255

    image = Image.fromarray(image_array, mode="RGB")
    mask = Image.fromarray(mask_array, mode="L")

    return image, mask


def test_horizontal_flip_is_applied_to_image_and_mask():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=1.0,
        vertical_flip_prob=0.0,
        rotation_prob=0.0,
    )

    augmented_image, augmented_mask = augmentation(
        image,
        mask,
    )

    expected_image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    expected_mask = mask.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    assert np.array_equal(
        np.asarray(augmented_image),
        np.asarray(expected_image),
    )

    assert np.array_equal(
        np.asarray(augmented_mask),
        np.asarray(expected_mask),
    )


def test_vertical_flip_is_applied_to_image_and_mask():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=0.0,
        vertical_flip_prob=1.0,
        rotation_prob=0.0,
    )

    augmented_image, augmented_mask = augmentation(
        image,
        mask,
    )

    expected_image = image.transpose(
        Image.Transpose.FLIP_TOP_BOTTOM
    )

    expected_mask = mask.transpose(
        Image.Transpose.FLIP_TOP_BOTTOM
    )

    assert np.array_equal(
        np.asarray(augmented_image),
        np.asarray(expected_image),
    )

    assert np.array_equal(
        np.asarray(augmented_mask),
        np.asarray(expected_mask),
    )


def test_rotation_preserves_image_and_mask_size():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_prob=1.0,
        rotation_degrees=15.0,
    )

    augmented_image, augmented_mask = augmentation(
        image,
        mask,
    )

    assert augmented_image.size == image.size
    assert augmented_mask.size == mask.size


def test_mask_remains_binary_after_rotation():
    image, mask = create_test_image_and_mask()

    augmentation = SegmentationAugmentation(
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_prob=1.0,
        rotation_degrees=15.0,
    )

    _, augmented_mask = augmentation(
        image,
        mask,
    )

    mask_values = set(
        np.unique(np.asarray(augmented_mask))
    )

    assert mask_values.issubset({0, 255})