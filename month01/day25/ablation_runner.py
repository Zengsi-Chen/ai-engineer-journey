import time

import torch
import torch.nn as nn
import torch.optim as optim

from cifar10_data import (
    create_cifar10_loaders,
)

from cpu_config import (
    configure_cpu,
)

from experiment_config import (
    CPUExperimentConfig,
)

from resnet18 import ResNet18

from resnet_ablation import (
    NoSkipResNet18,
)

from training_utils import (
    train_one_epoch,
    evaluate,
)

from experiment_results import (
    EpochResult,
    ExperimentResult,
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

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    print(
        f"\n{'=' * 50}"
    )

    print(
        f"Experiment: {name}"
    )

    print(
        f"{'=' * 50}"
    )

    best_validation_accuracy = 0.0

    epoch_results = []

    total_start = time.perf_counter()

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

        validation_loss, validation_accuracy = (
            evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )
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

        if validation_accuracy > (
            best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

        print(
            f"\nEpoch "
            f"{epoch + 1}/"
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

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        f"\nBest Validation Accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Training Time: "
        f"{total_time:.2f} seconds"
    )

    return ExperimentResult(
        name=name,
        epoch_results=epoch_results,
        best_validation_accuracy=(
            best_validation_accuracy
        ),
        training_time=total_time,
    )


def main():

    config = CPUExperimentConfig()

    configure_cpu(
        num_threads=config.num_threads
    )

    torch.manual_seed(
        config.seed
    )

    train_loader, validation_loader, _ = (
        create_cifar10_loaders(
            batch_size=config.batch_size,
            train_samples=config.train_samples,
            validation_samples=config.validation_samples,
            seed=config.seed,
            num_workers=config.num_workers,
        )
    )

    standard_model = ResNet18()

    standard_result = run_experiment(
        model=standard_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        name="Standard ResNet-18",
    )

    torch.manual_seed(
        config.seed
    )

    no_skip_model = NoSkipResNet18()

    no_skip_result = run_experiment(
        model=no_skip_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        name="No-Skip ResNet-18",
    )

    print(
        f"\n{'=' * 50}"
    )

    print(
        "FINAL ABLATION RESULT"
    )

    print(
        f"{'=' * 50}"
    )

    print(
        f"\nStandard ResNet-18:"
    )

    print(
        f"Best Val Accuracy: "
        f"{standard_result.best_validation_accuracy:.4f}"
    )

    print(
        f"Time: "
        f"{standard_result.training_time:.2f}s"
    )

    print(
        f"\nNo-Skip ResNet-18:"
    )

    print(
        f"Best Val Accuracy: "
        f"{no_skip_result.best_validation_accuracy:.4f}"
    )

    print(
        f"Time: "
        f"{no_skip_result.training_time:.2f}s"
    )


if __name__ == "__main__":
    main()