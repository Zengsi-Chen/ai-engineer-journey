import numpy as np

from model import LinearModel
from loss import mse_loss
from gradient import linear_gradients
from optimizer import SGD


X = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])

model = LinearModel()
optimizer = SGD(learning_rate=0.01)


for epoch in range(1000):

    # Prediction
    y_pred = model.forward(X)

    # Loss
    loss = mse_loss(y_pred, y)

    # Gradients
    dW, db = linear_gradients(X, y_pred, y)

    # Update
    optimizer.step(model, dW, db)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Loss: {loss:.6f}, "
            f"W: {model.W:.6f}, "
            f"b: {model.b:.6f}"
        )


print("\nFinal parameters:")
print("W:", model.W)
print("b:", model.b)

X_new = np.array([5, 10])
predictions = model.forward(X_new)

print("Predictions:", predictions)