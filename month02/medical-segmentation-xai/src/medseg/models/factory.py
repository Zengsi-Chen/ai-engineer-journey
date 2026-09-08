from torch import nn

from medseg.models.unet import UNet


def create_model(
    name: str,
    in_channels: int = 3,
    out_channels: int = 1,
    features: tuple[int, ...] = (32, 64, 128, 256),
) -> nn.Module:
    name = name.lower()

    if name == "unet":
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )

    raise ValueError(f"Unsupported model: {name}")