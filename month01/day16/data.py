import torch
from torch.utils.data import Dataset, random_split


class RegressionDataset(Dataset):

    def __init__(self, n_samples=30):

        torch.manual_seed(42)

        self.x = torch.linspace(
            0,
            10,
            n_samples
        ).reshape(-1, 1)

        noise = torch.randn(
            n_samples,
            1
        ) * 3.0

        self.y = 2 * self.x + 1 + noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def create_datasets(
    n_samples=30,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):

    dataset = RegressionDataset(
        n_samples=n_samples
    )

    train_size = int(
        n_samples * train_ratio
    )

    val_size = int(
        n_samples * val_ratio
    )

    test_size = (
        n_samples
        - train_size
        - val_size
    )

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )