import torch
import torch.nn as nn


class IdentityResidualBlock(
    nn.Module
):
    def __init__(
        self,
        channels,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x,
    ):

        identity = x

        out = self.conv1(x)

        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity

        return out