import torch

from medseg.evaluation.metrics import (
    sigmoid_predictions,
    dice_score,
    iou_score,
)


def test_sigmoid_predictions_shape():
    logits = torch.randn(2, 1, 32, 32)

    predictions = sigmoid_predictions(logits)

    assert predictions.shape == logits.shape


def test_sigmoid_predictions_binary():
    logits = torch.randn(2, 1, 32, 32)

    predictions = sigmoid_predictions(logits)

    unique_values = torch.unique(predictions)

    assert set(unique_values.tolist()).issubset({0.0, 1.0})


def test_perfect_prediction_dice():
    targets = torch.zeros(1, 1, 32, 32)
    targets[:, :, 10:20, 10:20] = 1.0

    logits = torch.full_like(targets, -10.0)
    logits[:, :, 10:20, 10:20] = 10.0

    score = dice_score(
        logits,
        targets,
    )

    assert score.item() > 0.99


def test_perfect_prediction_iou():
    targets = torch.zeros(1, 1, 32, 32)
    targets[:, :, 10:20, 10:20] = 1.0

    logits = torch.full_like(targets, -10.0)
    logits[:, :, 10:20, 10:20] = 10.0

    score = iou_score(
        logits,
        targets,
    )

    assert score.item() > 0.99


def test_completely_wrong_prediction():
    targets = torch.ones(1, 1, 32, 32)

    logits = torch.full_like(
        targets,
        -10.0,
    )

    dice = dice_score(
        logits,
        targets,
    )

    iou = iou_score(
        logits,
        targets,
    )

    assert dice.item() < 0.01
    assert iou.item() < 0.01


def test_metrics_are_finite():
    logits = torch.randn(2, 1, 32, 32)
    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    dice = dice_score(
        logits,
        targets,
    )

    iou = iou_score(
        logits,
        targets,
    )

    assert torch.isfinite(dice)
    assert torch.isfinite(iou)


def test_metrics_are_in_valid_range():
    logits = torch.randn(2, 1, 32, 32)
    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    dice = dice_score(
        logits,
        targets,
    )

    iou = iou_score(
        logits,
        targets,
    )

    assert 0.0 <= dice.item() <= 1.0
    assert 0.0 <= iou.item() <= 1.0


def test_empty_mask_perfect_prediction():
    targets = torch.zeros(
        1,
        1,
        32,
        32,
    )

    logits = torch.full_like(
        targets,
        -10.0,
    )

    dice = dice_score(
        logits,
        targets,
    )

    iou = iou_score(
        logits,
        targets,
    )

    assert torch.isfinite(dice)
    assert torch.isfinite(iou)

    assert dice.item() > 0.99
    assert iou.item() > 0.99