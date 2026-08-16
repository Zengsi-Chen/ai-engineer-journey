import torch
import torch.nn as nn


# -------------------------
# 1. Dataset
# -------------------------

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y_true = torch.tensor([
    [5.0],
    [8.0],
    [11.0],
    [14.0]
])


# -------------------------
# 2. Model
# -------------------------

class LinearModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearModel()


# -------------------------
# 3. Loss
# -------------------------

criterion = nn.MSELoss()


# -------------------------
# 4. Optimizer
# -------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# -------------------------
# 5. Training Loop
# -------------------------

epochs = 1000

for epoch in range(epochs):

    # Forward
    y_pred = model(x)

    # Loss
    loss = criterion(y_pred, y_true)

    # Clear old gradients
    optimizer.zero_grad()

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"Loss = {loss.item():.6f}"
        )


# -------------------------
# 6. Final Result
# -------------------------

print("\nFinal Parameters:")

for name, param in model.named_parameters():
    print(name, param.data)


# -------------------------
# 7. Prediction
# -------------------------

with torch.no_grad():

    prediction = model(x)

    print("\nPrediction:")
    print(prediction)