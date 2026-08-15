import numpy as np


class LinearModel:

    def __init__(self, input_size):
        self.W = np.random.randn(input_size)
        self.b = 0.0

    def forward(self, X):
        return X @ self.W + self.b