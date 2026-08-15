import numpy as np


def predict(X, W, b):
    return X @ W + b


def train(X, y, learning_rate=0.01, epochs=1000):

    W = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(epochs):

        prediction = predict(X, W, b)

        error = prediction - y
        loss = np.mean(error ** 2)

        dW = (2 / len(X)) * (X.T @ error)
        db = (2 / len(X)) * np.sum(error)

        W = W - learning_rate * dW
        b = b - learning_rate * db

    return W, b


def main():

    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
    ])

    y = np.array([
        8.0,
        7.0,
        15.0,
        14.0,
    ])

    W, b = train(X, y)

    print("W:", W)
    print("b:", b)

    new_X = np.array([
        [5, 6],
        [6, 5],
    ])

    predictions = predict(new_X, W, b)

    print("Predictions:", predictions)


if __name__ == "__main__":
    main()