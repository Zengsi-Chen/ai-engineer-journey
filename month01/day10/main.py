import numpy as np

from dataset import Dataset, DataLoader
from model import LinearModel
from loss import MSELoss
from gradient import LinearGradient
from optimizer import SGD


# =====================
# 1. Prepare data
# =====================

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
], dtype=float)

y = np.array([
    6,
    9,
    12,
    15,
    18,
    21,
    24,
    27
], dtype=float)


# =====================
# 2. Dataset
# =====================

dataset = Dataset(X, y)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# =====================
# 3. Model
# =====================

model = LinearModel(
    input_size=2
)


# =====================
# 4. Loss
# =====================

loss_fn = MSELoss()


# =====================
# 5. Gradient
# =====================

gradient_fn = LinearGradient()


# =====================
# 6. Optimizer
# =====================

optimizer = SGD(
    learning_rate=0.01
)


# =====================
# 7. Training
# =====================

epochs = 100

for epoch in range(epochs):

    epoch_loss = 0.0

    for X_batch, y_batch in dataloader:

        # Forward
        prediction = model.forward(X_batch)

        # Loss
        loss = loss_fn.forward(
            prediction,
            y_batch
        )

        # Backward
        dW, db = gradient_fn.backward(
            X_batch,
            prediction,
            y_batch
        )

        # Update
        optimizer.step(
            model,
            dW,
            db
        )

        epoch_loss += loss

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch}, "
            f"Loss: {epoch_loss:.4f}"
        )


# =====================
# 8. Results
# =====================

print("\nW:", model.W)
print("b:", model.b)

print(
    "Predictions:",
    model.forward(X)
)