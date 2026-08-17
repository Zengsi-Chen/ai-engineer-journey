import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):

    def __init__(self, n_samples=30):

        torch.manual_seed(42)

        self.x = torch.linspace(0, 10, n_samples).reshape(-1, 1)

        noise = torch.randn(n_samples, 1) * 3.0

        self.y = 2 * self.x + 1 + noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]