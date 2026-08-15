import numpy as np


def mse_loss(y_pred, y):
    error = y_pred - y
    squared_error = error ** 2
    return np.mean(squared_error)

if __name__ == "__main__":
    y_pred = np.array([2, 4, 6])
    y = np.array([3, 5, 7])

    loss = mse_loss(y_pred, y)

    print("Loss:", loss)