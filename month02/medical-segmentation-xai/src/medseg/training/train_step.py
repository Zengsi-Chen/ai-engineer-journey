import torch
from torch import nn


def train_one_batch(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    model.train()

    optimizer.zero_grad()

    logits = model(images)

    loss = criterion(logits, masks)

    loss.backward()

    optimizer.step()

    return {
        "loss": float(loss.item()),
    }