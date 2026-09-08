from pathlib import Path

import torch
from torch import nn


class BestModelCheckpoint:
    def __init__(
        self,
        path: Path,
        monitor: str = "val_dice",
        mode: str = "max",
    ) -> None:
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode

        if mode not in {"max", "min"}:
            raise ValueError(
                "mode must be either 'max' or 'min'"
            )

        if mode == "max":
            self.best_value = float("-inf")
        else:
            self.best_value = float("inf")

        self.best_epoch = None

    def is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value

        return value < self.best_value

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> bool:

        current_value = metrics[self.monitor]

        if not self.is_better(current_value):
            return False

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            self.path,
        )

        self.best_value = current_value
        self.best_epoch = epoch

        return True