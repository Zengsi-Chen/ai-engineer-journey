from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from domain_shift import (
    DomainShiftConfig,
    create_deterministic_domain_shift_transform,
    )

@dataclass
class TargetDomainData:

    name: str
    train_loader: DataLoader
    val_loader: DataLoader


def create_target_domain_data(
    root,
    name,
    config,
    batch_size=128,
    num_workers=0,
    normalize_mean=None,
    normalize_std=None,
    ):

    train_transform = (
    create_deterministic_domain_shift_transform(
        config=config,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        )
    )


    val_config = DomainShiftConfig(
        brightness=config.brightness,
        contrast=config.contrast,
        grayscale=config.grayscale,
        noise_std=0.0,
    )

    val_transform = (
        create_deterministic_domain_shift_transform(
            config=val_config,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    )

    train_dataset = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return TargetDomainData(
        name=name,
        train_loader=train_loader,
        val_loader=val_loader,
    )

