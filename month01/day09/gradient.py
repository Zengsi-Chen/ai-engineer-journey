import numpy as np


def linear_gradients(X, y_pred, y):
    error = y_pred - y

    dW = np.mean(2 * X * error)
    db = np.mean(2 * error)

    return dW, db

if __name__ == "__main__":
    X = np.array([1, 2, 3, 4])
    y_pred = np.array([0, 0, 0, 0])
    y = np.array([3, 5, 7, 9])

    dW, db = linear_gradients(X, y_pred, y)

    print("dW:", dW)
    print("db:", db)