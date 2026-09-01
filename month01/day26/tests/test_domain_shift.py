import torch

from PIL import Image
import numpy as np

from domain_shift import (
    DomainShiftConfig,
    GaussianNoise,
    create_domain_shift_transform,
)

def test_gaussian_noise_shape():

    noise = GaussianNoise(
        std=0.1
    )

    x = torch.rand(
        3,
        32,
        32,
    )

    y = noise(x)

    assert y.shape == x.shape

def test_gaussian_noise_range():

    noise = GaussianNoise(
        std=1.0
    )

    x = torch.rand(
        3,
        32,
        32,
    )

    y = noise(x)

    assert torch.all(y >= 0)
    assert torch.all(y <= 1)

def test_grayscale_keeps_three_channels():

    image = Image.fromarray(
        np.zeros(
            (32, 32, 3),
            dtype=np.uint8,
        )
    )

    config = DomainShiftConfig(
        grayscale=True
    )

    transform = create_domain_shift_transform(
        config
    )

    result = transform(image)

    assert result.shape == (
        3,
        32,
        32,
    )

def test_no_shift():

    config = DomainShiftConfig()

    transform = create_domain_shift_transform(
        config
    )

    image = Image.fromarray(
        np.zeros(
            (32, 32, 3),
            dtype=np.uint8,
        )
    )

    result = transform(image)

    assert result.shape == (
        3,
        32,
        32,
    )

    assert result.dtype == torch.float32