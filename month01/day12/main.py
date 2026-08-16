import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. Data
# =========================

X = torch.arange(1, 51).float().reshape(-1, 1)
y = 3 * X + 2


# =========================
# 2. Dataset
# =========================

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


dataset = MyDataset(X, y)


# =========================
# 3. DataLoader
# =========================

loader = DataLoader(
    dataset,
    batch_size=5,
    shuffle=True
)


# =========================
# 4. Model
# =========================

model = nn.Linear(
    in_features=1,
    out_features=1
)


# =========================
# 5. Loss
# =========================

loss_fn = nn.MSELoss()


# =========================
# 6. Optimizer
# =========================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.0001
)


# =========================
# 7. Training
# =========================

epochs = 100

loss_history = []

for epoch in range(epochs):

    total_loss = 0.0

    for X_batch, y_batch in loader:

        prediction = model(X_batch)

        loss = loss_fn(
            prediction,
            y_batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    loss_history.append(average_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {average_loss:.4f}"
        )

print(f"\nFinal Loss: {loss_history[-1]:.4f}")

# =========================
# 8. Parameters
# =========================

print("\nWeight:", model.weight.item())
print("Bias:", model.bias.item())


# =========================
# 9. Prediction
# =========================

test_x = torch.tensor([
    [10.0],
    [20.0],
    [30.0]
])

prediction = model(test_x)

print("\nPrediction:")
print(prediction)