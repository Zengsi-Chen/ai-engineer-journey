import torch
from torch import nn

from checkpoint import CheckpointManager
from dataset import create_dataloaders
from model import CIFAR10CNN
from train import train_one_epoch, evaluate

from plots import (
    plot_training_history,
    plot_accuracy_history,
)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=64,
        )
    )

    model = CIFAR10CNN(
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_manager = CheckpointManager(
        directory="artifacts"
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0

    epochs = 5

    for epoch in range(1, epochs + 1):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint_manager.save_best(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
            )

    print()
    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    plot_training_history(
        history
    )

    plot_accuracy_history(
        history
    )


if __name__ == "__main__":
    main()