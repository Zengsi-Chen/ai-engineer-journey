from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_DIR = Path("data")

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


def create_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    return train_transform, eval_transform

def create_dataloaders(
    batch_size: int = 64,
    val_ratio: float = 0.1,
):
    train_transform, eval_transform = create_transforms()

    train_dataset_full = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset_full = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform,
    )

    dataset_size = len(train_dataset_full)

    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(
        train_dataset_full,
        train_indices,
    )

    val_dataset = Subset(
        val_dataset_full,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader