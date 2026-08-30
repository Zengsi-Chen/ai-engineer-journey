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

from training_utils import (
    train_one_epoch,
    evaluate,
)

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

def calculate_accuracy(
    predictions,
    targets,
):

    predicted_classes = torch.argmax(
        predictions,
        dim=1,
    )

    correct = (
        predicted_classes
        == targets
    ).sum().item()

    total = targets.size(0)

    return correct, total


def main():

    config = CPUExperimentConfig()

    configure_cpu(
        num_threads=config.num_threads
    )

    torch.manual_seed(
        config.seed
    )

    start_time = time.perf_counter()

    device = get_device()

    print(
        f"\nUsing device: {device}"
    )

    train_loader, validation_loader, test_loader = (
        create_cifar10_loaders(
            batch_size=config.batch_size,
            train_samples=config.train_samples,
            validation_samples=config.validation_samples,
            seed=config.seed,
            num_workers=config.num_workers,
        )
    )

    model = ResNet18(
        num_classes=10
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    epochs = config.epochs

    best_validation_accuracy = 0.0

    for epoch in range(config.epochs):

        epoch_start_time = time.perf_counter()

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

        epoch_end_time = time.perf_counter()

        epoch_time = (
            epoch_end_time
            - epoch_start_time
        )

        print(
            f"\nEpoch "
            f"{epoch + 1}/{epochs}"
        )

        print(
            f"Time: "
            f"{epoch_time:.2f} seconds"
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

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                model.state_dict(),
                "best_resnet18.pth",
            )

            print(
                "Best model saved"
            )

    print(
        "\nLoading best model..."
    )

    model.load_state_dict(
        torch.load(
            "best_resnet18.pth",
            map_location=device,
        )
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        "\n=== Final Test Results ==="
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\nTotal Time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Total Time: "
        f"{total_time / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()