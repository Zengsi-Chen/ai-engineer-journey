import numpy as np


class LinearModel:
    def __init__(self):
        self.W = 0.0
        self.b = 0.0

    def forward(self, X):
        return X * self.W + self.b

if __name__ == "__main__":
    model = LinearModel()

    X = np.array([1, 2, 3, 4])

    predictions = model.forward(X)

    print("W:", model.W)
    print("b:", model.b)
    print("Predictions:", predictions)