import torch

from torch.utils.data import Dataset


class BinaryDataset(Dataset):

    def __init__(self):

        self.x = (
            torch.arange(
                1,
                101
            ).reshape(-1, 1)
            / 100
        )

        self.y = (
            self.x >= 0.5
        ).float()

    def __len__(self):

        return len(self.x)

    def __getitem__(self, index):

        return (
            self.x[index],
            self.y[index]
        )


from torch.utils.data import DataLoader


def create_binary_dataloaders(
    batch_size=32
):

    dataset = BinaryDataset()

    train_size = int(
        len(dataset) * 0.8
    )

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = (
        torch.utils.data.random_split(
            dataset,
            [train_size, val_size]
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader