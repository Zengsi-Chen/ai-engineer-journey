import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks.

    Conv → BatchNorm → ReLU
    Conv → BatchNorm → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block:
    DoubleConv → MaxPool
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.conv(x)

        pooled = self.pool(features)

        return features, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block:
    Upsample → concatenate skip connection → DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        x = self.conv(x)

        return x


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (
            32,
            64,
            128,
            256,
        ),
    ) -> None:

        super().__init__()

        self.enc1 = EncoderBlock(
            in_channels,
            features[0],
        )

        self.enc2 = EncoderBlock(
            features[0],
            features[1],
        )

        self.enc3 = EncoderBlock(
            features[1],
            features[2],
        )

        self.enc4 = EncoderBlock(
            features[2],
            features[3],
        )

        self.bottleneck = DoubleConv(
            features[3],
            features[3] * 2,
        )

        self.dec4 = DecoderBlock(
            features[3] * 2,
            features[3],
            features[3],
        )

        self.dec3 = DecoderBlock(
            features[3],
            features[2],
            features[2],
        )

        self.dec2 = DecoderBlock(
            features[2],
            features[1],
            features[1],
        )

        self.dec1 = DecoderBlock(
            features[1],
            features[0],
            features[0],
        )

        self.final_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = self.final_conv(x)

        return x