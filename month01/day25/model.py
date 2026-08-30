import torch
import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(8 * 8 * 8, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class DeepPlainCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(8 * 8 * 8, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class VeryDeepPlainCNN(nn.Module):
    def __init__(self, num_layers=20):
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                1,
                8,
                kernel_size=3,
                padding=1,
            )
        )

        layers.append(nn.Sigmoid())

        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    8,
                    8,
                    kernel_size=3,
                    padding=1,
                )
            )

            layers.append(nn.Sigmoid())

        layers.append(
            nn.Conv2d(
                8,
                8,
                kernel_size=3,
                padding=1,
            )
        )

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(
            8 * 8 * 8,
            2,
        )

    def forward(self, x):
        x = self.features(x)

        x = x.flatten(1)

        return self.classifier(x)