from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from medseg.training.checkpoint import BestModelCheckpoint
from medseg.training.early_stopping import EarlyStopping
from medseg.training.epoch import train_epoch, validate_epoch


@dataclass
class TrainingResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_dice: float | None = None
    stopped_early: bool = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        checkpoint: BestModelCheckpoint | None = None,
        scheduler=None,
        early_stopping: EarlyStopping | None = None,
        max_epochs: int = 10,
        threshold: float = 0.5,
    ) -> None:

        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.checkpoint = checkpoint
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.max_epochs = max_epochs
        self.threshold = threshold


    def fit(self) -> TrainingResult:
        self.model.to(self.device)

        history: list[dict[str, float]] = []
        stopped_early = False

        for epoch in range(1, self.max_epochs + 1):

            train_loss = train_epoch(
                model=self.model,
                loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
            )

            val_result = validate_epoch(
                model=self.model,
                loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                threshold=self.threshold,
            )

            val_loss = val_result["loss"]
            val_dice = val_result["dice"]
            val_iou = val_result["iou"]

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }

            if self.checkpoint is not None:
                saved = self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                )

                if saved:
                    print(
                        f"  -> New best model saved "
                        f"(Val Dice: {val_dice:.6f})"
                    )

            if self.scheduler is not None:
                self.scheduler.step(val_dice)

            current_lr = self.optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "learning_rate": current_lr,
                }
            )

            print(
                f"Epoch {epoch}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Dice: {val_dice:.6f} | "
                f"Val IoU: {val_iou:.6f} | "
                f"LR: {current_lr:.6g}"
            )

            if self.early_stopping is not None:
                should_stop = self.early_stopping.step(
                    value=val_dice,
                    epoch=epoch,
                )

                if should_stop:
                    stopped_early = True

                    print(
                        f"  -> Early stopping triggered "
                        f"(best epoch: "
                        f"{self.early_stopping.best_epoch}, "
                        f"best Val Dice: "
                        f"{self.early_stopping.best_value:.6f})"
                    )

                    break

        best_epoch = (
            self.checkpoint.best_epoch
            if self.checkpoint is not None
            else None
        )

        best_val_dice = (
            self.checkpoint.best_value
            if self.checkpoint is not None
            else None
        )

        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_val_dice=best_val_dice,
            stopped_early=stopped_early,
        )