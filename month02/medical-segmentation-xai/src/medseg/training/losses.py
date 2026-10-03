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


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if alpha < 0:
            raise ValueError(
                "alpha must be non-negative"
            )

        if beta < 0:
            raise ValueError(
                "beta must be non-negative"
            )

        if alpha + beta == 0:
            raise ValueError(
                "At least one of alpha or beta must be positive"
            )

        if smooth <= 0:
            raise ValueError(
                "smooth must be positive"
            )

        self.alpha = alpha
        self.beta = beta
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

        true_positive = (
            probabilities * targets
        ).sum(dim=1)

        false_positive = (
            probabilities * (1.0 - targets)
        ).sum(dim=1)

        false_negative = (
            (1.0 - probabilities) * targets
        ).sum(dim=1)

        tversky = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        return (1.0 - tversky).mean()
    

class BoundaryLoss(nn.Module):
    """
    Boundary-weighted binary cross-entropy loss.

    Pixels near the target boundary receive a larger loss weight.
    """

    def __init__(
        self,
        boundary_weight: float = 3.0,
    ) -> None:
        super().__init__()

        if boundary_weight < 1.0:
            raise ValueError(
                "boundary_weight must be >= 1.0"
            )

        self.boundary_weight = boundary_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        if logits.shape != targets.shape:
            raise ValueError(
                "logits and targets must have the same shape"
            )

        if logits.ndim != 4:
            raise ValueError(
                "logits and targets must have shape [B, C, H, W]"
            )

        targets = targets.float()

        # Detect target boundary using local 3x3 neighborhoods.
        padded = nn.functional.pad(
            targets,
            (1, 1, 1, 1),
            mode="replicate",
        )

        neighbors = [
            padded[:, :, 0:-2, 0:-2],
            padded[:, :, 0:-2, 1:-1],
            padded[:, :, 0:-2, 2:],
            padded[:, :, 1:-1, 0:-2],
            padded[:, :, 1:-1, 2:],
            padded[:, :, 2:, 0:-2],
            padded[:, :, 2:, 1:-1],
            padded[:, :, 2:, 2:],
        ]

        neighbor_stack = torch.stack(
            neighbors,
            dim=0,
        )

        local_min = neighbor_stack.min(dim=0).values
        local_max = neighbor_stack.max(dim=0).values

        boundary = (
            (local_max != local_min)
            .float()
        )

        weights = torch.ones_like(targets)

        weights = weights + (
            self.boundary_weight - 1.0
        ) * boundary

        pixel_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        weighted_loss = (
            pixel_loss * weights
        )

        return weighted_loss.mean()