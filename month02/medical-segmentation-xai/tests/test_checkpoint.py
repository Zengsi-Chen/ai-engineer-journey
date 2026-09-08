from pathlib import Path

import torch

from medseg.models.factory import create_model
from medseg.training.checkpoint import BestModelCheckpoint
from medseg.training.optimizer import create_optimizer


def create_test_model():
    return create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )


def test_checkpoint_saves_best_model(tmp_path):
    model = create_test_model()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    checkpoint = BestModelCheckpoint(
        path=checkpoint_path,
        monitor="val_dice",
        mode="max",
    )

    saved = checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={
            "train_loss": 0.5,
            "val_loss": 0.4,
            "val_dice": 0.60,
            "val_iou": 0.45,
        },
    )

    assert saved is True
    assert checkpoint_path.exists()
    assert checkpoint.best_epoch == 1
    assert checkpoint.best_value == 0.60


def test_checkpoint_does_not_save_worse_model(tmp_path):
    model = create_test_model()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    checkpoint = BestModelCheckpoint(
        path=checkpoint_path,
        monitor="val_dice",
        mode="max",
    )

    first_saved = checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={
            "val_dice": 0.70,
        },
    )

    second_saved = checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=2,
        metrics={
            "val_dice": 0.65,
        },
    )

    assert first_saved is True
    assert second_saved is False

    assert checkpoint.best_epoch == 1
    assert checkpoint.best_value == 0.70


def test_checkpoint_saves_improved_model(tmp_path):
    model = create_test_model()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    checkpoint = BestModelCheckpoint(
        path=checkpoint_path,
        monitor="val_dice",
        mode="max",
    )

    checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={
            "val_dice": 0.60,
        },
    )

    saved = checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=2,
        metrics={
            "val_dice": 0.75,
        },
    )

    assert saved is True
    assert checkpoint.best_epoch == 2
    assert checkpoint.best_value == 0.75


def test_checkpoint_can_be_loaded(tmp_path):
    model = create_test_model()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    checkpoint = BestModelCheckpoint(
        path=checkpoint_path,
        monitor="val_dice",
        mode="max",
    )

    checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics={
            "val_dice": 0.80,
        },
    )

    data = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert data["epoch"] == 3
    assert "model_state_dict" in data
    assert "optimizer_state_dict" in data
    assert "metrics" in data
    assert data["metrics"]["val_dice"] == 0.80