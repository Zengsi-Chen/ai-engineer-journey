from __future__ import annotations

import torch
from torch import Tensor


def compute_segmentation_target(
    logits: Tensor,
    threshold: float = 0.55,
    target_mode: str = "predicted_foreground",
) -> Tensor:
    """
    Convert segmentation logits into a scalar explanation target.

    Args:
        logits: Model output with shape [B, 1, H, W].
        threshold: Probability threshold used to define foreground.
        target_mode:
            "predicted_foreground":
                Sum probabilities only inside the predicted foreground.
            "soft_foreground":
                Sum all foreground probabilities without hard thresholding.

    Returns:
        Scalar target score for the whole batch.

    Raises:
        ValueError:
            If logits has an invalid shape or target_mode is unknown.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "logits must have shape [B, 1, H, W], "
            f"but received {tuple(logits.shape)}"
        )

    if target_mode not in {
        "predicted_foreground",
        "soft_foreground",
    }:
        raise ValueError(
            "target_mode must be either "
            "'predicted_foreground' or 'soft_foreground'"
        )

    probabilities = torch.sigmoid(logits)

    if target_mode == "soft_foreground":
        return probabilities.sum()

    predicted_mask = (probabilities >= threshold).float()

    return (probabilities * predicted_mask).sum()


def compute_input_saliency(
    model: torch.nn.Module,
    image: Tensor,
    threshold: float = 0.55,
    target_mode: str = "predicted_foreground",
) -> Tensor:
    """
    Compute an input-gradient saliency map for segmentation.

    Args:
        model: Trained segmentation model.
        image: Input tensor with shape [1, 3, H, W].
        threshold: Probability threshold for the predicted foreground.

    Returns:
        Normalized saliency map with shape [H, W] and values in [0, 1].

    Raises:
        ValueError: If image does not have shape [1, 3, H, W].
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(
            "image must have shape [1, 3, H, W], "
            f"but received {tuple(image.shape)}"
        )

    was_training = model.training
    model.eval()

    input_image = image.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)

    logits = model(input_image)

    target_score = compute_segmentation_target(
        logits=logits,
        threshold=threshold,
        target_mode=target_mode,
    )

    target_score.backward()

    gradients = input_image.grad

    if gradients is None:
        raise RuntimeError("Input gradients were not computed.")

    saliency = gradients.abs().max(dim=1).values.squeeze(0)

    saliency_min = saliency.min()
    saliency_max = saliency.max()

    denominator = saliency_max - saliency_min

    if denominator > 0:
        saliency = (saliency - saliency_min) / denominator
    else:
        saliency = torch.zeros_like(saliency)

    if was_training:
        model.train()

    return saliency.detach()