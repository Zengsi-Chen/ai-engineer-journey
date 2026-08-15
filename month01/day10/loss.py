import numpy as np


class MSELoss:

    def forward(self, prediction, target):
        return np.mean(
            (prediction - target) ** 2
        )