import numpy as np

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

Y = np.array([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])





W1 = np.random.randn(1, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.01

learning_rate = 0.01

for epoch in range(1000):

    # Forward
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Y_pred = A1 @ W2 + b2

    # Loss
    loss = np.mean((Y_pred - Y) ** 2)

    # Backward
    dY_pred = 2 * (Y_pred - Y) / len(X)

    dW2 = A1.T @ dY_pred
    db2 = np.sum(dY_pred, axis=0, keepdims=True)

    dA1 = dY_pred @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

Z1 = X @ W1 + b1
A1 = np.maximum(0, Z1)
Y_pred = A1 @ W2 + b2

print("Predictions:")
print(Y_pred)

print("Actual:")
print(Y)