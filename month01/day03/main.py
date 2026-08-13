import numpy as np


def linear_layer(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    return X @ W + b


def main():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=float)

    W = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)

    b = np.array([1, 1], dtype=float)

    Y = linear_layer(X, W, b)

    print("Input X:")
    print(X)

    print("\nWeights W:")
    print(W)

    print("\nBias b:")
    print(b)

    print("\nOutput Y:")
    print(Y)

    print("\nShapes:")
    print("X:", X.shape)
    print("W:", W.shape)
    print("b:", b.shape)
    print("Y:", Y.shape)


if __name__ == "__main__":
    main()