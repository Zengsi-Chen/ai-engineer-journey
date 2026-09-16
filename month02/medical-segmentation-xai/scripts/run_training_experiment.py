import json
import random
from pathlib import Path

import numpy as np
import torch
import json

from medseg.data.dataloader import create_segmentation_dataloader
from medseg.models.factory import create_model
from medseg.training.checkpoint import BestModelCheckpoint
from medseg.training.early_stopping import EarlyStopping
from medseg.training.losses import TverskyLoss
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    EXPERIMENT_NAME = "tversky_04_06"

    seed = 42
    device = torch.device("cpu")
    image_size = 256
    batch_size = 4
    num_workers = 0
    epochs = 10

    AUGMENTATION_MODE = "none"

    EXPERIMENT_DIR = Path("artifacts/day39") / EXPERIMENT_NAME
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_PATH = EXPERIMENT_DIR / "best_model.pt"
    RESULTS_PATH = EXPERIMENT_DIR / "training_results.json"
    CONFIG_PATH = EXPERIMENT_DIR / "config.json"

    train_loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augmentation=AUGMENTATION_MODE,
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

    criterion = TverskyLoss(
        alpha=0.4,
        beta=0.6,
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

    experiment_config = {
        "experiment_name": EXPERIMENT_NAME,
        "seed": seed,
        "device": str(device),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "augmentation": AUGMENTATION_MODE,
        "model": {
            "name": "unet",
            "in_channels": 3,
            "out_channels": 1,
            "features": [16, 32, 64, 128],
        },
        "loss": {
            "name": "TverskyLoss",
            "alpha": 0.4,
            "beta": 0.6,
            "smooth": 1.0,
        },
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
        "threshold": 0.5,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)


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

    results = {
        "experiment_name": EXPERIMENT_NAME,
        "best_epoch": result.best_epoch,
        "best_val_dice": result.best_val_dice,
        "stopped_early": result.stopped_early,
        "history": result.history,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Results:    {RESULTS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Config:     {CONFIG_PATH}")
    print("=" * 80)

    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Epochs completed: {len(result.history)}")
    print(f"Best Epoch: {result.best_epoch}")
    print(f"Best Val Dice: {result.best_val_dice:.6f}")
    print(f"Stopped Early: {result.stopped_early}")
    print("=" * 60)


if __name__ == "__main__":
    main()