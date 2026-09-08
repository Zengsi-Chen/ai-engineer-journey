import json
import random
from pathlib import Path

import numpy as np
import torch

from medseg.data.dataloader import create_segmentation_dataloader
from medseg.models.factory import create_model
from medseg.training.checkpoint import BestModelCheckpoint
from medseg.training.early_stopping import EarlyStopping
from medseg.training.losses import BCEDiceLoss
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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "best_model.pt"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "day34_trainer_experiment.json"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    seed = 42
    set_seed(seed)

    device = torch.device("cpu")

    image_size = 256
    batch_size = 4
    num_workers = 0
    epochs = 10

    train_loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
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
        path=CHECKPOINT_PATH,
        monitor="val_dice",
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=3,
        min_delta=0.001,
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
        threshold=0.5,
    )

    result = trainer.fit()

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": {
            "name": "day34_trainer",
            "seed": seed,
            "device": str(device),
        },
        "model": {
            "name": "unet",
            "features": [16, 32, 64, 128],
        },
        "training": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "name": "plateau",
            "monitor": "val_dice",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 1e-6,
        },
        "early_stopping": {
            "monitor": "val_dice",
            "patience": 3,
            "min_delta": 0.001,
        },
        "result": {
            "best_epoch": result.best_epoch,
            "best_val_dice": result.best_val_dice,
            "stopped_early": result.stopped_early,
            "epochs_completed": len(result.history),
        },
        "history": result.history,
        "checkpoint": str(CHECKPOINT_PATH),
    }

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Epochs completed: {len(result.history)}")
    print(f"Best Epoch: {result.best_epoch}")
    print(f"Best Val Dice: {result.best_val_dice:.6f}")
    print(f"Stopped Early: {result.stopped_early}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Experiment: {RESULT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()