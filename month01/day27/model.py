import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import torch

from resnet import BasicBlock


class TransferLearningResNet(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)



class ResNet18(nn.Module):

    def __init__(
        self,
        num_classes=10,
    ):
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(
            out_channels=64,
            num_blocks=2,
            stride=1,
        )

        self.layer2 = self._make_layer(
            out_channels=128,
            num_blocks=2,
            stride=2,
        )

        self.layer3 = self._make_layer(
            out_channels=256,
            num_blocks=2,
            stride=2,
        )

        self.layer4 = self._make_layer(
            out_channels=512,
            num_blocks=2,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            512,
            num_classes,
        )

    def _make_layer(
        self,
        out_channels,
        num_blocks,
        stride,
    ):

        layers = []

        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )

        self.in_channels = out_channels

        for _ in range(
            1,
            num_blocks,
        ):

            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(
            *layers
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.classifier(x)

        return x