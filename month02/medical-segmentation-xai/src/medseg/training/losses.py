import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(
            probabilities.shape[0], -1
        )

        targets = targets.reshape(
            targets.shape[0], -1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            (2.0 * intersection + self.smooth)
            / (denominator + self.smooth)
        )

        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if bce_weight < 0:
            raise ValueError(
                "bce_weight must be non-negative"
            )

        if dice_weight < 0:
            raise ValueError(
                "dice_weight must be non-negative"
            )

        if bce_weight + dice_weight == 0:
            raise ValueError(
                "At least one loss weight must be positive"
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = BCELoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )