import torch

from domain_shift import (
    DomainShiftConfig,
    )

from target_domain_data import (
    create_target_domain_data,
    )

def test_create_target_domain_data(
    tmp_path,
    ):


    config = DomainShiftConfig(
        brightness=0.2,
        contrast=0.2,
        grayscale=False,
        noise_std=0.02,
    )

    data = create_target_domain_data(
        root=str(tmp_path),
        name="test_domain",
        config=config,
        batch_size=4,
    )

    assert data.name == "test_domain"

    assert (
        len(data.train_loader.dataset)
        == 50000
    )

    assert (
        len(data.val_loader.dataset)
        == 10000
    )


def test_target_domain_batch_shape(
    tmp_path,
    ):


    config = DomainShiftConfig(
        brightness=0.0,
        contrast=0.0,
        grayscale=True,
        noise_std=0.0,
    )

    data = create_target_domain_data(
        root=str(tmp_path),
        name="grayscale_domain",
        config=config,
        batch_size=4,
    )

    inputs, targets = next(
        iter(data.train_loader)
    )

    assert isinstance(
        inputs,
        torch.Tensor,
    )

    assert inputs.shape == (
        4,
        3,
        32,
        32,
    )

    assert targets.shape == (
        4,
    )
