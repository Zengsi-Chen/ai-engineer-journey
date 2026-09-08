import pytest
import torch

from medseg.training.losses import (
    BCELoss,
    DiceLoss,
    BCEDiceLoss,
)


def test_bce_loss_returns_scalar():

    logits = torch.randn(
        2, 1, 32, 32
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    criterion = BCELoss()

    loss = criterion(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_dice_loss_returns_scalar():

    logits = torch.randn(
        2, 1, 32, 32
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    criterion = DiceLoss()

    loss = criterion(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0


def test_dice_loss_perfect_prediction():

    targets = torch.ones(
        1, 1, 8, 8
    )

    logits = torch.full(
        (1, 1, 8, 8),
        10.0,
    )

    criterion = DiceLoss()

    loss = criterion(
        logits,
        targets,
    )

    assert loss.item() < 0.001


def test_bce_dice_loss_returns_scalar():

    logits = torch.randn(
        2, 1, 32, 32
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    criterion = BCEDiceLoss()

    loss = criterion(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_invalid_loss_weights():

    with pytest.raises(ValueError):

        BCEDiceLoss(
            bce_weight=0.0,
            dice_weight=0.0,
        )


def test_negative_loss_weight():

    with pytest.raises(ValueError):

        BCEDiceLoss(
            bce_weight=-1.0,
            dice_weight=1.0,
        )


def test_combined_loss_supports_backward():

    logits = torch.randn(
        2,
        1,
        32,
        32,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    criterion = BCEDiceLoss()

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dice_loss_empty_mask_is_finite():

    logits = torch.full(
        (1, 1, 32, 32),
        -10.0,
    )

    targets = torch.zeros(
        1,
        1,
        32,
        32,
    )

    criterion = DiceLoss()

    loss = criterion(
        logits,
        targets,
    )

    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0


