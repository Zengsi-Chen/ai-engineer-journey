from pathlib import Path

import torch

from medseg.data.dataloader import create_segmentation_dataloader
from medseg.models.factory import create_model
from medseg.training.epoch import train_epoch, validate_epoch
from medseg.training.losses import BCEDiceLoss
from medseg.training.optimizer import create_optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "split_manifest.csv"
)


def create_test_components():
    train_loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    val_loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=256,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    criterion = BCEDiceLoss()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    device = torch.device("cpu")

    return (
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device,
    )


def test_train_epoch_returns_finite_loss():
    (
        train_loader,
        _,
        model,
        criterion,
        optimizer,
        device,
    ) = create_test_components()

    loss = train_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert torch.isfinite(torch.tensor(loss))


def test_validate_epoch_returns_metrics():
    (
        _,
        val_loader,
        model,
        criterion,
        optimizer,
        device,
    ) = create_test_components()

    result = validate_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    assert isinstance(result, dict)

    assert "loss" in result
    assert "dice" in result
    assert "iou" in result

    assert result["loss"] >= 0.0
    assert 0.0 <= result["dice"] <= 1.0
    assert 0.0 <= result["iou"] <= 1.0

    assert torch.isfinite(
        torch.tensor(result["loss"])
    )

    assert torch.isfinite(
        torch.tensor(result["dice"])
    )

    assert torch.isfinite(
        torch.tensor(result["iou"])
    )


def test_validate_epoch_does_not_update_parameters():
    (
        _,
        val_loader,
        model,
        criterion,
        optimizer,
        device,
    ) = create_test_components()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    validate_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    after = list(model.parameters())

    for before_parameter, after_parameter in zip(before, after):
        assert torch.equal(before_parameter, after_parameter)