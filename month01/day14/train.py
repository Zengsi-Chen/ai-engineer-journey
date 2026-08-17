import torch

from model import LinearModel
from checkpoint import (
    save_checkpoint,
    load_checkpoint
)


def train():

    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]]
    )

    y = torch.tensor(
        [[3.0], [5.0], [7.0], [9.0]]
    )

    model = LinearModel()

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    start_epoch = 0

    checkpoint_path = "checkpoint.pth"

    try:

        start_epoch, previous_loss = load_checkpoint(
            model,
            optimizer,
            checkpoint_path
        )

        print(
            f"Checkpoint loaded. "
            f"Resume from epoch {start_epoch}"
        )

    except FileNotFoundError:

        print("No checkpoint found. Starting from scratch.")

    epochs = 200

    for epoch in range(
        start_epoch,
        epochs
    ):

        prediction = model(x)

        loss = criterion(
            prediction,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {loss.item():.4f}"
            )

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                loss.item(),
                checkpoint_path
            )