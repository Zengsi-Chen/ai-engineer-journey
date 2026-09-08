import torch

from medseg.training.losses import (
    BCELoss,
    DiceLoss,
    BCEDiceLoss,
)


def evaluate_case(
    name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:

    bce = BCELoss()
    dice = DiceLoss()
    combined = BCEDiceLoss()

    bce_value = bce(logits, targets)
    dice_value = dice(logits, targets)
    combined_value = combined(logits, targets)

    probabilities = torch.sigmoid(logits)

    print("-" * 60)
    print(f"Case: {name}")
    print(f"Target foreground ratio: {targets.mean().item():.4f}")
    print(f"Prediction probability mean: {probabilities.mean().item():.4f}")
    print(f"BCE Loss: {bce_value.item():.6f}")
    print(f"Dice Loss: {dice_value.item():.6f}")
    print(f"BCE + Dice Loss: {combined_value.item():.6f}")


def main() -> None:

    print("=" * 60)
    print("Segmentation Loss Behavior Experiment")
    print("=" * 60)

    # --------------------------------------------------
    # Case A: Perfect foreground prediction
    # --------------------------------------------------

    targets = torch.ones(1, 1, 32, 32)

    logits = torch.full(
        (1, 1, 32, 32),
        10.0,
    )

    evaluate_case(
        name="Perfect foreground prediction",
        logits=logits,
        targets=targets,
    )

    # --------------------------------------------------
    # Case B: Completely wrong prediction
    # --------------------------------------------------

    targets = torch.ones(1, 1, 32, 32)

    logits = torch.full(
        (1, 1, 32, 32),
        -10.0,
    )

    evaluate_case(
        name="Completely wrong foreground prediction",
        logits=logits,
        targets=targets,
    )

    # --------------------------------------------------
    # Case C: Small foreground
    # --------------------------------------------------

    targets = torch.zeros(1, 1, 32, 32)

    targets[:, :, 14:18, 14:18] = 1.0

    logits = torch.full(
        (1, 1, 32, 32),
        -5.0,
    )

    logits[:, :, 14:18, 14:18] = 5.0

    evaluate_case(
        name="Small foreground",
        logits=logits,
        targets=targets,
    )

    # --------------------------------------------------
    # Case D: Empty mask
    # --------------------------------------------------

    targets = torch.zeros(1, 1, 32, 32)

    logits = torch.full(
        (1, 1, 32, 32),
        -10.0,
    )

    evaluate_case(
        name="Empty foreground",
        logits=logits,
        targets=targets,
    )

    print("=" * 60)


if __name__ == "__main__":
    main()