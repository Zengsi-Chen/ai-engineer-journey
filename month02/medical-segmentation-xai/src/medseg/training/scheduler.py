import torch
from torch import nn


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "plateau",
    mode: str = "max",
    factor: float = 0.5,
    patience: int = 2,
    min_lr: float = 1e-6,
):
    name = name.lower()

    if mode not in {"min", "max"}:
        raise ValueError("mode must be either 'min' or 'max'")

    if not 0 < factor < 1:
        raise ValueError("factor must be between 0 and 1")

    if patience < 0:
        raise ValueError("patience must be non-negative")

    if min_lr < 0:
        raise ValueError("min_lr must be non-negative")

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(f"Unsupported scheduler: {name}")