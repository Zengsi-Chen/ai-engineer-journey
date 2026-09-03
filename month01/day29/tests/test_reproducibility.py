import random

import numpy as np
import pytest
import torch

from src.reproducibility import (
    get_environment_metadata,
    get_reproducibility_metadata,
    set_seed,
)


def test_python_random_is_reproducible():
    set_seed(42)

    first_value = random.random()

    set_seed(42)

    second_value = random.random()

    assert first_value == second_value


def test_numpy_random_is_reproducible():
    set_seed(42)

    first_values = np.random.rand(5)

    set_seed(42)

    second_values = np.random.rand(5)

    assert np.array_equal(
        first_values,
        second_values,
    )


def test_torch_random_is_reproducible():
    set_seed(42)

    first_values = torch.rand(5)

    set_seed(42)

    second_values = torch.rand(5)

    assert torch.equal(
        first_values,
        second_values,
    )


def test_different_seeds_produce_different_values():
    set_seed(42)

    first_values = torch.rand(5)

    set_seed(123)

    second_values = torch.rand(5)

    assert not torch.equal(
        first_values,
        second_values,
    )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        "42",
        42.0,
        None,
        True,
        False,
    ],
)
    
def test_seed_must_be_integer(invalid_seed):
    with pytest.raises(
        TypeError,
        match="Seed must be an integer",
    ):
        set_seed(invalid_seed)


def test_environment_metadata_contains_required_fields():
    metadata = get_environment_metadata()

    required_fields = {
        "python_version",
        "pytorch_version",
        "platform",
        "device",
        "cuda_version",
        "cudnn_version",
    }

    assert required_fields.issubset(
        metadata.keys()
    )


def test_environment_metadata_device_is_valid():
    metadata = get_environment_metadata()

    assert metadata["device"] in {
        "cpu",
        "cuda",
    }


def test_reproducibility_metadata():
    seed = 42

    metadata = get_reproducibility_metadata(
        seed
    )

    assert metadata["seed"] == seed

    assert "environment" in metadata

    assert (
        metadata["environment"]["device"]
        in {"cpu", "cuda"}
    )