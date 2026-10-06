import json
import random
from pathlib import Path

import numpy as np
import torch

from medseg.data.dataloader import create_segmentation_dataloader
from medseg.models.factory import create_model
from medseg.training.checkpoint import BestModelCheckpoint
from medseg.training.early_stopping import EarlyStopping
from medseg.training.losses import (
    BCEDiceLoss,
    BoundaryLoss,
)
from medseg.training.optimizer import create_optimizer
from medseg.training.scheduler import create_scheduler
from medseg.training.trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "split_manifest.csv"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_experiment_loss(
    experiment_name: str,
    boundary_loss_weight: float,
):
    if experiment_name == "baseline":
        return BCEDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5,
        )

    if experiment_name == "boundary":
        return BoundaryAwareLoss(
            bce_weight=0.5,
            dice_weight=0.5,
            boundary_weight=3.0,
            boundary_loss_weight=boundary_loss_weight,
        )

    raise ValueError(
        f"Unknown experiment: {experiment_name}"
    )


class BoundaryAwareLoss(torch.nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 3.0,
        boundary_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()

        self.bce_dice = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        self.boundary = BoundaryLoss(
            boundary_weight=boundary_weight,
        )

        if boundary_loss_weight < 0:
            raise ValueError(
                "boundary_loss_weight must be non-negative"
            )

        self.boundary_loss_weight = boundary_loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        base_loss = self.bce_dice(
            logits,
            targets,
        )

        boundary_loss = self.boundary(
            logits,
            targets,
        )

        return (
            base_loss
            + self.boundary_loss_weight * boundary_loss
        )


def run_experiment(
    experiment_name: str,
    boundary_loss_weight: float,
) -> None:

    seed = 42
    set_seed(seed)

    device = torch.device("cpu")

    image_size = 256
    batch_size = 4
    num_workers = 0
    epochs = 10
    threshold = 0.55

    augmentation_mode = "none"

    experiment_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "day44"
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        experiment_dir
        / "best_model.pt"
    )

    results_path = (
        experiment_dir
        / "training_results.json"
    )

    config_path = (
        experiment_dir
        / "config.json"
    )

    train_loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augmentation=augmentation_mode,
    )

    val_loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augmentation="none",
    )

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    criterion_name = (
        "baseline"
        if boundary_loss_weight == 0.0
        else "boundary"
    )

    criterion = create_experiment_loss(
        criterion_name,
        boundary_loss_weight,
    )

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        name="plateau",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    checkpoint = BestModelCheckpoint(
        path=checkpoint_path,
        monitor="val_dice",
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=3,
        min_delta=0.001,
    )

    if experiment_name == "baseline":
        loss_config = {
            "name": "BCEDiceLoss",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        }

    else:
        loss_config = {
            "name": "BoundaryAwareLoss",
            "base_loss": "BCEDiceLoss",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "boundary_weight": 3.0,
            "boundary_loss_weight": boundary_loss_weight,
        }

    experiment_config = {
        "experiment_name": experiment_name,
        "seed": seed,
        "device": str(device),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "augmentation": augmentation_mode,
        "threshold": threshold,
        "model": {
            "name": "unet",
            "in_channels": 3,
            "out_channels": 1,
            "features": [16, 32, 64, 128],
        },
        "loss": loss_config,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "name": "plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 1e-6,
        },
        "early_stopping": {
            "monitor": "val_dice",
            "mode": "max",
            "patience": 3,
            "min_delta": 0.001,
        },
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment_config,
            f,
            indent=2,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint=checkpoint,
        scheduler=scheduler,
        early_stopping=early_stopping,
        max_epochs=epochs,
        threshold=threshold,
    )

    result = trainer.fit()

    results = {
        "experiment_name": experiment_name,
        "best_epoch": result.best_epoch,
        "best_val_dice": result.best_val_dice,
        "stopped_early": result.stopped_early,
        "history": result.history,
    }

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print(f"Day 44 Experiment: {experiment_name}")
    print(f"Results:    {results_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_path}")
    print("=" * 80)

    print(
        f"Best Epoch: {result.best_epoch}"
    )

    print(
        f"Best Val Dice: "
        f"{result.best_val_dice:.6f}"
    )

    print(
        f"Stopped Early: "
        f"{result.stopped_early}"
    )


def main() -> None:

    print("=" * 80)
    print("DAY 44 — Boundary Loss Weight Ablation")
    print("=" * 80)

    experiments = [
        ("baseline", 0.00),
        ("boundary_005", 0.05),
        ("boundary_010", 0.10),
        ("boundary_020", 0.20),
    ]

    for experiment_name, boundary_loss_weight in experiments:

        print()
        print("-" * 80)
        print(
            f"Experiment: {experiment_name} | "
            f"Boundary Loss Weight: {boundary_loss_weight}"
        )
        print("-" * 80)

        run_experiment(
            experiment_name,
            boundary_loss_weight,
        )


if __name__ == "__main__":
    main()