import torch

from dataset import create_dataloaders
from evaluate import evaluate
from model import (
    create_resnet18,
    get_trainable_parameters,
)
from train import train_one_epoch
from checkpoint import CheckpointManager


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, test_loader = create_dataloaders(
        batch_size=4,
        num_workers=0,
    )

    model = create_resnet18(
        num_classes=10,
        fine_tune=True,
    ).to(device)

    '''
    trainable_parameters = list(
        get_trainable_parameters(model)
    )

    trainable_count = sum(
        param.numl()
        for param in trainable_parameters 
    )

    print(
        "Trainable parameter count:",
        trainable_count,
    )

    print("\nTrainable parameters:")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    '''

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.layer4.parameters(),
                "lr": 1e-4,
            },
            {
                "params": model.fc.parameters(),
                "lr": 1e-3,
            },
        ]
    )

    print("\nLearning rates:")

    print(
        "Layer4 LR:",
        optimizer.param_groups[0]["lr"],
    )

    print(
        "FC LR:",
        optimizer.param_groups[1]["lr"],
    )

    checkpoint_manager = CheckpointManager("artifacts/checkpoints")

    epochs = 5

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    best_test_accuracy = 0.01
    best_epoch = 0

    for epoch in range(1, epochs + 1):


        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch

            checkpoint_manager.save_best_model(
                model=model,
                epoch=epoch,
                metric=test_accuracy,
            )

            print(
                f"  New best model! "
                f"Test Accuracy: {test_accuracy:.4f}"
            )

    print("\nTraining History:")
    print(history)

    print("\nBest Model:")
    print("Best Epoch:", best_epoch)
    print("Best Test Accuracy:", best_test_accuracy)
    print(
        "Best Model Path:",
        checkpoint_manager.best_model_path,
    )

    best_model = create_resnet18(
        num_classes=10
    ).to(device)

    checkpoint = checkpoint_manager.load_best_model(
        model=best_model,
        device=device,
    )

    print("\nLoaded checkpoint:")
    print("Epoch:", checkpoint["epoch"])
    print("Metric:", checkpoint["metric"])

    loaded_test_loss, loaded_test_accuracy = evaluate(
        model=best_model,
        dataloader=test_loader,
        device=device,
    )

    print(
        "Loaded model test accuracy:",
        loaded_test_accuracy,
    )

if __name__ == "__main__":
    main()