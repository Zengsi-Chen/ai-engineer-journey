import torch
from torch import nn


def create_optimizer(
    model: nn.Module,
    name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:

    name = name.lower()

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay must be non-negative"
        )

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {name}"
    )