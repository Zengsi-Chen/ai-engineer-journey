import torch
import torch.nn as nn

from medseg.models.unet import DoubleConv, EncoderBlock


class AttentionGate(nn.Module):
    """
    Attention gate for filtering encoder skip features.

    The decoder feature acts as a gating signal that helps
    suppress irrelevant spatial activations in the skip connection.
    """

    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        inter_channels: int,
    ) -> None:

        super().__init__()

        self.skip_projection = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        self.gating_projection = nn.Sequential(
            nn.Conv2d(
                gating_channels,
                inter_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        skip: torch.Tensor,
        gating: torch.Tensor,
    ) -> torch.Tensor:

        skip_projection = self.skip_projection(skip)
        gating_projection = self.gating_projection(gating)

        attention = self.attention(
            skip_projection + gating_projection
        )

        return skip * attention


class AttentionDecoderBlock(nn.Module):
    """
    Attention U-Net decoder block:

    Upsample → Attention Gate → concatenate → DoubleConv
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

        self.attention = AttentionGate(
            skip_channels=skip_channels,
            gating_channels=out_channels,
            inter_channels=max(out_channels // 2, 1),
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

        skip = self.attention(
            skip=skip,
            gating=x,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        x = self.conv(x)

        return x


class AttentionUNet(nn.Module):
    """
    U-Net with attention-gated skip connections.

    The encoder and decoder channel configuration matches
    the existing UNet implementation so that the experiment
    changes only the skip-connection mechanism.
    """

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

        self.dec4 = AttentionDecoderBlock(
            features[3] * 2,
            features[3],
            features[3],
        )

        self.dec3 = AttentionDecoderBlock(
            features[3],
            features[2],
            features[2],
        )

        self.dec2 = AttentionDecoderBlock(
            features[2],
            features[1],
            features[1],
        )

        self.dec1 = AttentionDecoderBlock(
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