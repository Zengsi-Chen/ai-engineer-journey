import numpy as np
import pytest

from src.data_split import DataSplitter


def create_dataset():
    X = np.arange(100).reshape(50, 2)

    y = np.array(
        [0] * 25 +
        [1] * 25
    )

    return X, y


def test_split_produces_expected_sizes():
    X, y = create_dataset()

    splitter = DataSplitter(
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = splitter.split(X, y)

    assert len(X_train) == 30
    assert len(X_val) == 10
    assert len(X_test) == 10

    assert len(y_train) == 30
    assert len(y_val) == 10
    assert len(y_test) == 10


def test_split_has_no_sample_overlap():
    X, y = create_dataset()

    splitter = DataSplitter(
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    (
        X_train,
        X_val,
        X_test,
        _,
        _,
        _,
    ) = splitter.split(X, y)

    train_rows = {tuple(row) for row in X_train}
    val_rows = {tuple(row) for row in X_val}
    test_rows = {tuple(row) for row in X_test}

    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)


def test_split_preserves_class_distribution():
    X, y = create_dataset()

    splitter = DataSplitter(
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    (
        _,
        _,
        _,
        y_train,
        y_val,
        y_test,
    ) = splitter.split(X, y)

    assert y_train.mean() == pytest.approx(0.5)
    assert y_val.mean() == pytest.approx(0.5)
    assert y_test.mean() == pytest.approx(0.5)


def test_invalid_split_sizes_raise_error():
    with pytest.raises(ValueError):
        DataSplitter(
            validation_size=0.6,
            test_size=0.5,
        )