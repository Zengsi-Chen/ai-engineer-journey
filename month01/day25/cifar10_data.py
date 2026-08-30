import torch

from torch.utils.data import (
    DataLoader,
    Subset,
)

from torchvision import datasets

from cifar10_augmentation import (
    create_train_transform,
    create_evaluation_transform,
)


def create_cifar10_loaders(
    batch_size=128,
    train_samples=2000,
    validation_samples=500,
    seed=42,
    num_workers=0,
):

    train_transform = (
        create_train_transform()
    )

    evaluation_transform = (
        create_evaluation_transform()
    )

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    validation_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=evaluation_transform,
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=evaluation_transform,
    )

    generator = torch.Generator()

    generator.manual_seed(seed)

    total_samples = (
        train_samples
        + validation_samples
    )

    indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    )[:total_samples]

    train_indices = indices[
        :train_samples
    ]

    validation_indices = indices[
        train_samples:
    ]

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )