import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

    def forward(self, x):

        return self.network(x)


class SimpleCNN(nn.Module):

    def __init__(
        self,
        input_channels=1,
        num_classes=10
    ):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Linear(
            16 * 14 * 14,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x