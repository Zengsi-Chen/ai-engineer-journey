import torch

from model import create_resnet18
from train import train_one_epoch
from evaluate import evaluate


def create_feature_extraction_experiment(config):
    model = create_resnet18(
        num_classes=config.num_classes,
        fine_tune=False,
    )

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=config.classifier_lr,
    )

    return model, optimizer


def create_fine_tuning_experiment(config):
    model = create_resnet18(
        num_classes=config.num_classes,
        fine_tune=True,
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.layer4.parameters(),
                "lr": config.backbone_lr,
            },
            {
                "params": model.fc.parameters(),
                "lr": config.classifier_lr,
            },
        ]
    )

    return model, optimizer


def count_trainable_parameters(model):
    return sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )


def run_experiment(
    model,
    optimizer,
    train_loader,
    test_loader,
    device,
    epochs,
):
    model = model.to(device)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    best_test_accuracy = 0.0
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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

    return {
        "history": history,
        "best_test_accuracy": best_test_accuracy,
        "best_epoch": best_epoch,
    }