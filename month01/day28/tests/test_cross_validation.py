import numpy as np
import pytest

from src.cross_validation import KFoldValidator


def test_kfold_generates_expected_number_of_folds():
    X = np.arange(20).reshape(10, 2)

    validator = KFoldValidator(
        n_splits=5,
        random_state=42,
    )

    folds = list(
        validator.split(X)
    )

    assert len(folds) == 5


def test_each_fold_has_no_index_overlap():
    X = np.arange(20).reshape(10, 2)

    validator = KFoldValidator(
        n_splits=5,
        random_state=42,
    )

    for train_indices, val_indices in validator.split(X):
        train_set = set(train_indices)
        val_set = set(val_indices)

        assert train_set.isdisjoint(val_set)


def test_every_sample_is_validation_once():
    X = np.arange(20).reshape(10, 2)

    validator = KFoldValidator(
        n_splits=5,
        random_state=42,
    )

    validation_indices = []

    for _, val_indices in validator.split(X):
        validation_indices.extend(val_indices)

    assert sorted(validation_indices) == list(range(len(X)))


def test_aggregate_scores():
    validator = KFoldValidator(
        n_splits=5,
    )

    result = validator.aggregate([
        0.80,
        0.90,
        0.85,
        0.95,
        0.90,
    ])

    assert result.fold_scores == [
        0.80,
        0.90,
        0.85,
        0.95,
        0.90,
    ]

    assert result.mean_score == pytest.approx(0.88)

    assert result.std_score == pytest.approx(
        np.std([
            0.80,
            0.90,
            0.85,
            0.95,
            0.90,
        ])
    )


def test_invalid_number_of_scores_raises_error():
    validator = KFoldValidator(
        n_splits=5,
    )

    with pytest.raises(ValueError):
        validator.aggregate([
            0.80,
            0.90,
        ])