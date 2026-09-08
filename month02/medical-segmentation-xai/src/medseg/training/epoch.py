import torch
from torch import nn
from torch.utils.data import DataLoader

from medseg.evaluation.metrics import dice_score, iou_score


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, masks)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader is empty")

    return total_loss / total_samples


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = criterion(logits, masks)

        dice = dice_score(
            logits,
            masks,
            threshold=threshold,
        )

        iou = iou_score(
            logits,
            masks,
            threshold=threshold,
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        total_iou += iou.item() * batch_size

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation loader is empty")

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }