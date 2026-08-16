import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import RegressionDataset
from model import NonlinearModel


# --------------------
# Dataset
# --------------------

dataset = RegressionDataset(30)

train_size = 15
val_size = 15

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# --------------------
# DataLoader
# --------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# --------------------
# Model
# --------------------

model = NonlinearModel()

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)


# --------------------
# Early Stopping
# --------------------

best_val_loss = float("inf")

patience = 30
counter = 0

epochs = 1000


# --------------------
# Training
# --------------------

for epoch in range(epochs):

    # TRAIN

    model.train()

    train_loss = 0.0

    for x, y in train_loader:

        prediction = model(x)

        loss = loss_fn(
            prediction,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # VALIDATION

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            prediction = model(x)

            loss = loss_fn(
                prediction,
                y
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # EARLY STOPPING

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

    else:

        counter += 1


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )


    if counter >= patience:

        print(
            f"Early stopping at epoch "
            f"{epoch + 1}"
        )

        break


# --------------------
# Restore Best Model
# --------------------

model.load_state_dict(
    torch.load("best_model.pth")
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)