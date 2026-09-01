import torch

from domain_shift import (
    DomainShiftConfig,
    create_deterministic_domain_shift_transform,
    )

def test_deterministic_domain_shift():


    config = DomainShiftConfig(
        brightness=0.1,
        contrast=0.2,
        grayscale=False,
    )

    transform = (
        create_deterministic_domain_shift_transform(
            config=config,
        )
    )

    image = torch.rand(
        3,
        32,
        32,
    )

    output1 = transform(
        image.permute(1, 2, 0).numpy()
    )

    output2 = transform(
        image.permute(1, 2, 0).numpy()
    )

    assert torch.equal(
        output1,
        output2,
    )


def test_different_domain_shift_changes_image():


    image = torch.rand(
        3,
        32,
        32,
    )

    config = DomainShiftConfig(
        brightness=0.3,
        contrast=0.0,
        grayscale=False,
    )

    transform = (
        create_deterministic_domain_shift_transform(
            config=config,
        )
    )

    shifted = transform(
        image.permute(1, 2, 0).numpy()
    )

    original = image

    assert not torch.equal(
        shifted,
        original,
    )

