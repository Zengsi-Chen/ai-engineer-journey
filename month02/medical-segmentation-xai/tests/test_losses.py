import pytest
import torch

from medseg.training.losses import (
    BCELoss,
    DiceLoss,
    BCEDiceLoss,
    TverskyLoss,
    BoundaryLoss,
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


def test_tversky_loss_returns_scalar():

    logits = torch.randn(
        2,
        1,
        32,
        32,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    criterion = TverskyLoss(
        alpha=0.3,
        beta=0.7,
    )

    loss = criterion(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_tversky_loss_perfect_prediction():

    targets = torch.ones(
        1,
        1,
        8,
        8,
    )

    logits = torch.full(
        (1, 1, 8, 8),
        10.0,
    )

    criterion = TverskyLoss(
        alpha=0.3,
        beta=0.7,
    )

    loss = criterion(
        logits,
        targets,
    )

    assert loss.item() < 0.001


def test_tversky_loss_invalid_parameters():

    with pytest.raises(ValueError):

        TverskyLoss(
            alpha=-0.1,
            beta=0.7,
        )

    with pytest.raises(ValueError):

        TverskyLoss(
            alpha=0.3,
            beta=-0.1,
        )

    with pytest.raises(ValueError):

        TverskyLoss(
            alpha=0.0,
            beta=0.0,
        )

    with pytest.raises(ValueError):

        TverskyLoss(
            alpha=0.3,
            beta=0.7,
            smooth=0.0,
        )


def test_tversky_loss_supports_backward():

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

    criterion = TverskyLoss(
        alpha=0.3,
        beta=0.7,
    )

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_tversky_loss_penalizes_false_negative_more_than_false_positive():

    targets = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0]]]]
    )

    # Baseline: correct prediction for both foreground pixels
    perfect_logits = torch.tensor(
        [[[[10.0, 10.0, -10.0, -10.0]]]]
    )

    # One foreground pixel is missed
    false_negative_logits = torch.tensor(
        [[[[10.0, -10.0, -10.0, -10.0]]]]
    )

    # One background pixel is incorrectly predicted as foreground
    false_positive_logits = torch.tensor(
        [[[[10.0, 10.0, 10.0, -10.0]]]]
    )

    criterion = TverskyLoss(
        alpha=0.3,
        beta=0.7,
    )

    perfect_loss = criterion(
        perfect_logits,
        targets,
    )

    false_negative_loss = criterion(
        false_negative_logits,
        targets,
    )

    false_positive_loss = criterion(
        false_positive_logits,
        targets,
    )

    false_negative_penalty = (
        false_negative_loss - perfect_loss
    )

    false_positive_penalty = (
        false_positive_loss - perfect_loss
    )

    assert false_negative_penalty > false_positive_penalty


def test_boundary_loss_scalar_and_finite():
    loss_fn = BoundaryLoss()

    logits = torch.randn(
        2, 1, 16, 16,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 16, 16),
    ).float()

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_boundary_loss_perfect_prediction():
    loss_fn = BoundaryLoss()

    targets = torch.zeros(
        1, 1, 8, 8,
    )

    targets[:, :, 2:6, 2:6] = 1.0

    logits = torch.where(
        targets > 0,
        torch.full_like(targets, 20.0),
        torch.full_like(targets, -20.0),
    )

    loss = loss_fn(logits, targets)

    assert loss.item() < 1e-6


def test_boundary_loss_backward():
    loss_fn = BoundaryLoss()

    logits = torch.randn(
        2, 1, 16, 16,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 16, 16),
    ).float()

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_boundary_loss_rejects_shape_mismatch():
    loss_fn = BoundaryLoss()

    logits = torch.randn(1, 1, 8, 8)
    targets = torch.randn(1, 1, 16, 16)

    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_boundary_loss_rejects_invalid_dimensions():
    loss_fn = BoundaryLoss()

    logits = torch.randn(1, 8, 8)
    targets = torch.randn(1, 8, 8)

    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_boundary_weight_validation():
    with pytest.raises(ValueError):
        BoundaryLoss(boundary_weight=0.5)

    BoundaryLoss(boundary_weight=1.0)
    BoundaryLoss(boundary_weight=3.0)


