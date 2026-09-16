import torch
import torch.nn as nn

from medseg.xai.saliency import (
    compute_segmentation_target,
    compute_input_saliency,
)


class TinySegmentationModel(nn.Module):
    """Minimal segmentation model for testing XAI."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


def test_compute_segmentation_target_returns_scalar():
    logits = torch.ones(1, 1, 4, 4)

    target = compute_segmentation_target(
        logits,
        threshold=0.55,
    )

    assert target.ndim == 0
    assert target.item() > 0


def test_compute_segmentation_target_uses_threshold():
    logits = torch.tensor(
        [
            [
                [
                    [2.0, -2.0],
                    [2.0, -2.0],
                ]
            ]
        ]
    )

    target = compute_segmentation_target(
        logits,
        threshold=0.55,
    )

    probabilities = torch.sigmoid(logits)

    expected = probabilities[
        probabilities >= 0.55
    ].sum()

    assert torch.allclose(target, expected)


def test_compute_input_saliency_shape_and_range():
    model = TinySegmentationModel()

    image = torch.rand(
        1,
        3,
        16,
        16,
    )

    saliency = compute_input_saliency(
        model=model,
        image=image,
        threshold=0.55,
    )

    assert saliency.shape == (16, 16)

    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0


def test_compute_input_saliency_produces_nonzero_gradient():
    model = TinySegmentationModel()

    with torch.no_grad():
        model.conv.weight.fill_(1.0)
        model.conv.bias.fill_(1.0)

    image = torch.rand(
        1,
        3,
        16,
        16,
    )

    saliency = compute_input_saliency(
        model=model,
        image=image,
        threshold=0.55,
    )

    assert torch.isfinite(saliency).all()
    assert saliency.max() > 0.0


def test_soft_foreground_saliency_works_for_complete_miss():
    model = TinySegmentationModel()

    with torch.no_grad():
        model.conv.weight.fill_(0.1)
        model.conv.bias.fill_(-2.0)

    image = torch.rand(
        1,
        3,
        16,
        16,
    )

    saliency = compute_input_saliency(
        model=model,
        image=image,
        threshold=0.55,
        target_mode="soft_foreground",
    )

    assert torch.isfinite(saliency).all()
    assert saliency.shape == (16, 16)
    assert saliency.max() > 0.0