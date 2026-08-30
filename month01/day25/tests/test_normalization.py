import torch

from cifar10_augmentation import (
    CIFAR10_MEAN,
    CIFAR10_STD,
)


def test_cifar10_normalization_constants():

    assert len(
        CIFAR10_MEAN
    ) == 3

    assert len(
        CIFAR10_STD
    ) == 3


def test_normalization_formula():

    image = torch.tensor(
        [
            [
                [
                    0.5
                ]
            ]
        ]
    )

    mean = 0.4

    std = 0.2

    normalized = (
        image - mean
    ) / std

    expected = torch.tensor(
        [
            [
                [
                    0.5
                ]
            ]
        ]
    )

    assert torch.allclose(
        normalized,
        expected,
    )