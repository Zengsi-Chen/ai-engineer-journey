import numpy as np
import pytest

from src.preprocessing import SafePreprocessor


def test_transform_before_fit_raises_error():
    preprocessor = SafePreprocessor()

    X = np.array([
        [1.0],
        [2.0],
    ])

    with pytest.raises(RuntimeError):
        preprocessor.transform(X)


def test_fit_transform_train_data():
    preprocessor = SafePreprocessor()

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    X_train_scaled = preprocessor.fit_transform(X_train)

    assert preprocessor.is_fitted is True

    np.testing.assert_allclose(
        X_train_scaled.mean(axis=0),
        np.array([0.0]),
    )


def test_validation_uses_train_statistics():
    preprocessor = SafePreprocessor()

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    X_val = np.array([
        [100.0],
        [200.0],
    ])

    preprocessor.fit(X_train)

    X_val_scaled = preprocessor.transform(X_val)

    assert X_val_scaled.mean() > 1.0