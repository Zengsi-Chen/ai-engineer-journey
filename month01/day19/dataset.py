import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)


class ClassificationDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def create_dataset():
    """
    Create the same dataset used in Day 18.
    """

    x = torch.arange(
        1,
        101,
        dtype=torch.float32
    ).reshape(-1, 1)

    x = x / 100.0

    y = (x >= 0.5).float()

    dataset = ClassificationDataset(
        x,
        y
    )

    return dataset


def split_dataset(
    dataset,
    train_ratio=0.8,
    seed=42
):
    """
    Split dataset using the same logic as Day 18.
    """

    train_size = int(
        train_ratio * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )

    generator = torch.Generator().manual_seed(
        seed
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_dataset, val_dataset


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=False
):
    """
    Create a DataLoader.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )