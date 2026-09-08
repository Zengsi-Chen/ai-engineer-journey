import torch

from medseg.models.unet import UNet
from medseg.models.factory import create_model


def test_unet_output_shape():

    model = UNet(
        in_channels=3,
        out_channels=1,
    )

    x = torch.randn(
        2,
        3,
        256,
        256,
    )

    y = model(x)

    assert y.shape == (
        2,
        1,
        256,
        256,
    )


def test_unet_output_dtype():

    model = UNet(
        in_channels=3,
        out_channels=1,
    )

    x = torch.randn(
        2,
        3,
        256,
        256,
    )

    y = model(x)

    assert y.dtype == torch.float32


def test_model_factory():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
    )

    assert isinstance(model, UNet)


def test_model_factory_invalid_name():

    try:
        create_model("unknown")
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for unknown model"
    )


def test_unet_parameter_count():

    model = UNet(
        in_channels=3,
        out_channels=1,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert total_parameters > 0


def test_unet_forward_is_finite():

    model = UNet(
        in_channels=3,
        out_channels=1,
    )

    x = torch.randn(
        1,
        3,
        256,
        256,
    )

    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all()


def test_unet_small_configuration():
    model = UNet(
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 256, 256)
    assert torch.isfinite(y).all()


def test_factory_supports_custom_features():
    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 256, 256)
    assert torch.isfinite(y).all()