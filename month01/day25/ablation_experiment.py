import time

import torch
import torch.nn as nn
import torch.optim as optim

from experiment_results import (
    EpochResult,
    ExperimentResult,
)

from training_utils import (
    train_one_epoch,
    evaluate,
)

def run_experiment(
    model,
    train_loader,
    validation_loader,
    config,
    name,
):

    device = torch.device(
        "cpu"
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    epoch_results = []

    best_validation_accuracy = 0.0

    total_start_time = (
        time.perf_counter()
    )

    for epoch in range(
        config.epochs
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        (
            validation_loss,
            validation_accuracy,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        epoch_result = EpochResult(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )

        epoch_results.append(
            epoch_result
        )

        best_validation_accuracy = max(
            best_validation_accuracy,
            validation_accuracy,
        )

        print(
            f"\n{name} "
            f"Epoch {epoch + 1}/"
            f"{config.epochs}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

    total_training_time = (
        time.perf_counter()
        - total_start_time
    )

    return ExperimentResult(
        name=name,
        epoch_results=epoch_results,
        best_validation_accuracy=(
            best_validation_accuracy
        ),
        training_time=total_training_time,
    )