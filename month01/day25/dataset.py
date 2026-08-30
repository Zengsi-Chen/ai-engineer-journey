import torch
from torch.utils.data import TensorDataset, DataLoader


def create_toy_dataloader(
    n_samples=512,
    batch_size=32,
    seed=42,
):
    generator = torch.Generator().manual_seed(seed)

    x = torch.randn(
        n_samples,
        1,
        8,
        8,
        generator=generator,
    )

    scores = x.mean(dim=(1, 2, 3))

    y = (scores > 0).long()

    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )