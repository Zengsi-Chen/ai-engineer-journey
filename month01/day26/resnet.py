import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
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

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity

        out = self.relu(out)

        return out

class DeepResidualCNN(nn.Module):
    def __init__(
        self,
        num_blocks=4,
        channels=8,
    ):
        super().__init__()

        self.stem = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.classifier = nn.Linear(
            channels * 8 * 8,
            2,
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.blocks(x)

        x = x.flatten(1)

        return self.classifier(x)


class ProjectionResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if (
            in_channels != out_channels
            or stride != 1
        ):

            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity

        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
        self,
        in_channels,
        base_channels,
        stride=1,
    ):
        super().__init__()

        out_channels = (
            base_channels
            * self.expansion
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            base_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            base_channels
        )

        self.conv3 = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(
            out_channels
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)

        return out