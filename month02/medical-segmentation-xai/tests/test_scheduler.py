import torch
import pytest

from medseg.models.factory import create_model
from medseg.training.optimizer import create_optimizer
from medseg.training.scheduler import create_scheduler


def test_create_plateau_scheduler():
    model = create_model(
        name="unet",
        features=(8, 16, 32, 64),
    )

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        name="plateau",
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )


def test_scheduler_reduces_learning_rate():
    model = create_model(
        name="unet",
        features=(8, 16, 32, 64),
    )

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        name="plateau",
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    initial_lr = optimizer.param_groups[0]["lr"]

    scheduler.step(0.5)
    scheduler.step(0.5)
    scheduler.step(0.5)

    current_lr = optimizer.param_groups[0]["lr"]

    assert current_lr < initial_lr


def test_invalid_scheduler_name():
    model = create_model(
        name="unet",
        features=(8, 16, 32, 64),
    )

    optimizer = create_optimizer(model)

    with pytest.raises(ValueError):
        create_scheduler(
            optimizer=optimizer,
            name="invalid",
        )


def test_invalid_scheduler_mode():
    model = create_model(
        name="unet",
        features=(8, 16, 32, 64),
    )

    optimizer = create_optimizer(model)

    with pytest.raises(ValueError):
        create_scheduler(
            optimizer=optimizer,
            mode="invalid",
        )