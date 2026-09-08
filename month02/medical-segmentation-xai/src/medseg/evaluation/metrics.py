import torch


def sigmoid_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    return predictions


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    predictions = sigmoid_predictions(
        logits,
        threshold=threshold,
    )

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )

    targets = targets.reshape(
        targets.shape[0],
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    denominator = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return dice.mean()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    predictions = sigmoid_predictions(
        logits,
        threshold=threshold,
    )

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )

    targets = targets.reshape(
        targets.shape[0],
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()