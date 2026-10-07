import torch

from medseg.models.attention_unet import AttentionUNet
from medseg.models.factory import create_model


def test_attention_unet_output_shape():
    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    x = torch.randn(2, 3, 256, 256)
    y = model(x)

    assert y.shape == (2, 1, 256, 256)


def test_attention_unet_factory():
    model = create_model(
        name="attention_unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    assert isinstance(model, AttentionUNet)


def test_attention_unet_parameter_count():
    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    parameter_count = sum(p.numel() for p in model.parameters())

    assert parameter_count == 1_964_941