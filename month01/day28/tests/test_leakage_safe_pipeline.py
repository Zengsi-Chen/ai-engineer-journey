import numpy as np
import pytest

from src.leakage_safe_pipeline import (
    LeakageSafeImagePipeline,
)


def create_test_data():
    rng = np.random.default_rng(42)

    images = rng.normal(
        size=(50, 1, 8, 8)
    ).astype(np.float32)

    labels = np.array(
        [0, 1] * 25
    )

    groups = np.repeat(
        np.arange(10),
        5,
    )

    return images, labels, groups


def test_sample_pipeline():
    images, labels, groups = (
        create_test_data()
    )

    pipeline = LeakageSafeImagePipeline(
        split_unit="sample",
        validation_size=0.2,
        test_size=0.2,
    )

    result = pipeline.prepare(
        images,
        labels,
    )

    assert result["train"]["images"].shape[0] == 30
    assert result["validation"]["images"].shape[0] == 10
    assert result["test"]["images"].shape[0] == 10

    assert "groups" not in result["train"]


def test_group_pipeline():
    images, labels, groups = (
        create_test_data()
    )

    pipeline = LeakageSafeImagePipeline(
        split_unit="group",
        validation_size=0.2,
        test_size=0.2,
    )

    result = pipeline.prepare(
        images,
        labels,
        groups=groups,
    )

    train_groups = set(
        result["train"]["groups"]
    )

    val_groups = set(
        result["validation"]["groups"]
    )

    test_groups = set(
        result["test"]["groups"]
    )

    assert train_groups.isdisjoint(
        val_groups
    )

    assert train_groups.isdisjoint(
        test_groups
    )

    assert val_groups.isdisjoint(
        test_groups
    )


def test_group_pipeline_requires_groups():
    images, labels, groups = (
        create_test_data()
    )

    pipeline = LeakageSafeImagePipeline(
        split_unit="group"
    )

    with pytest.raises(ValueError):
        pipeline.prepare(
            images,
            labels,
        )


def test_invalid_split_unit():
    with pytest.raises(ValueError):
        LeakageSafeImagePipeline(
            split_unit="invalid"
        )