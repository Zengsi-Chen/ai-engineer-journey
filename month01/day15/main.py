import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import RegressionDataset
from model import NonlinearModel
from early_stopping import EarlyStopping
from checkpoint import (
    save_checkpoint,
    load_checkpoint
)

# --------------------
# Dataset
# --------------------

dataset = RegressionDataset(30)

train_size = 20
val_size = 10

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

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10
)

early_stopping = EarlyStopping(
    patience=20,
    min_delta=0.001,
    path="best_model.pth"
)

# --------------------
# First time resume = False, then after we have checkpoint.oth resume = True
# --------------------

resume = True
start_epoch = 0

if resume:

    (
        start_epoch,
        best_val_loss,
        counter
    ) = load_checkpoint(
        "checkpoint.pth",
        model,
        optimizer,
        scheduler
    )

    early_stopping.best_val_loss = best_val_loss
    early_stopping.counter = counter

    print(
        f"Resuming from epoch {start_epoch}"
    )


# --------------------
# Scheduler and Early Stopping
# --------------------

num_epochs = 1000

for epoch in range(start_epoch, num_epochs):

    # ----------------
    # Training
    # ----------------
    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ----------------
    # Validation
    # ----------------
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # ----------------
    # Scheduler
    # ----------------
    scheduler.step(val_loss)

    # ----------------
    # Early Stopping
    # ----------------
    early_stopping(
        val_loss,
        model
    )

    # ----------------
    # Checkpoint
    # ----------------

    save_checkpoint(
        "checkpoint.pth",
        model,
        optimizer,
        scheduler,
        epoch,
        early_stopping.best_val_loss,
        early_stopping.counter
    )

    # ----------------
    # Simulating interruption
    # ----------------
    current_epoch = epoch + 1

    print(
        f"Epoch [{current_epoch}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"LR: {current_lr:.6f}"
    )

    if current_epoch == 200:
        print("Simulating interruption...")
        break


    # ----------------
    # Logging
    # ----------------
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"LR: {current_lr:.6f}"
    )

   
    if early_stopping.early_stop:

        print(
            f"Early stopping at epoch {epoch+1}"
        )

        break

print("Loading best model...")

model.load_state_dict(
    torch.load("best_model.pth")
)

print(
    f"Best validation loss: "
    f"{early_stopping.best_val_loss:.4f}"
)
   
