from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = Path("data")


def create_dataloaders(
    batch_size=64,
    num_workers=0,
):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader