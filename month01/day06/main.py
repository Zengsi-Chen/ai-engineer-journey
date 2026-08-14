import numpy as np


def predict(x, w, b):
    return w * x + b


def compute_loss(prediction, y):
    return np.mean((prediction - y) ** 2)


def compute_gradients(x, prediction, y):
    gradient_w = np.mean(2 * (prediction - y) * x)
    gradient_b = np.mean(2 * (prediction - y))

    return gradient_w, gradient_b


def train(x, y, learning_rate, epochs):
    w = 0.0
    b = 0.0

    for epoch in range(epochs):
        prediction = predict(x, w, b)

        loss = compute_loss(prediction, y)

        gradient_w, gradient_b = compute_gradients(
            x, prediction, y
        )

        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b

        if epoch % 10 == 0:
            print(
                "epoch:", epoch,
                "loss:", loss,
                "w:", w,
                "b:", b
            )

    return w, b


def main():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 7, 9])

    learning_rates = [0.001, 0.1, 1.0]

    for learning_rate in learning_rates:
        print()
        print("Learning rate:", learning_rate)

        w, b = train(
            x,
            y,
            learning_rate,
            100
        )

        print("Final w:", w)
        print("Final b:", b)


if __name__ == "__main__":
    main()