import numpy as np
import pytest

from src.normalization import TrainOnlyNormalizer


def test_transform_before_fit_raises_error():
    normalizer = TrainOnlyNormalizer()

    images = np.ones(
        (2, 1, 2, 2),
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError):
        normalizer.transform(images)


def test_fit_calculates_channel_statistics():
    normalizer = TrainOnlyNormalizer()

    images = np.array([
        [[[1.0, 3.0],
          [5.0, 7.0]]],

        [[[2.0, 4.0],
          [6.0, 8.0]]],
    ])

    normalizer.fit(images)

    expected_mean = images.mean(
        axis=(0, 2, 3),
        keepdims=True,
    )

    expected_std = images.std(
        axis=(0, 2, 3),
        keepdims=True,
    )

    np.testing.assert_allclose(
        normalizer.mean,
        expected_mean,
    )

    np.testing.assert_allclose(
        normalizer.std,
        expected_std,
    )


def test_validation_uses_train_statistics():
    normalizer = TrainOnlyNormalizer()

    train_images = np.ones(
        (2, 1, 2, 2),
        dtype=np.float32,
    ) * 10

    val_images = np.ones(
        (2, 1, 2, 2),
        dtype=np.float32,
    ) * 100

    normalizer.fit(train_images)

    val_normalized = normalizer.transform(
        val_images,
    )

    assert val_normalized.mean() > 1.0


def test_normalizer_statistics_come_from_training_only():
    train_images = np.ones(
        (4, 1, 2, 2),
        dtype=np.float32,
    ) * 10.0

    val_images = np.ones(
        (4, 1, 2, 2),
        dtype=np.float32,
    ) * 1000.0

    normalizer = TrainOnlyNormalizer()

    normalizer.fit(train_images)

    transformed_val = normalizer.transform(
        val_images
    )

    # Mean must be based on training data.
    assert np.isclose(
        normalizer.mean.item(),
        10.0,
    )

    # Validation data must NOT influence fitted mean.
    assert not np.isclose(
        normalizer.mean.item(),
        505.0,
    )

    # Validation data is transformed using train statistics.
    assert transformed_val.mean() > 100.0