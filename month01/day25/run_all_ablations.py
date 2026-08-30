import torch

from cifar10_data import (
    create_cifar10_loaders,
)

from cpu_config import (
    configure_cpu,
)

from experiment_config import (
    CPUExperimentConfig,
)

from resnet18 import (
    ResNet18,
)

from resnet_ablation import (
    NoSkipResNet18,
    NoBNResNet18,
)

from ablation_experiment import (
    run_experiment,
)

from experiment_results import (
    save_experiment_results,
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

    experiments = [
        (
            "Standard ResNet-18",
            ResNet18,
        ),
        (
            "No-Skip ResNet-18",
            NoSkipResNet18,
        ),
        (
            "No-BN ResNet-18",
            NoBNResNet18,
        ),
    ]

    results = []

    for name, model_class in experiments:

        print()

        print("=" * 60)

        print(
            f"Starting Experiment: {name}"
        )

        print("=" * 60)

        torch.manual_seed(
            config.seed
        )

        model = model_class()

        result = run_experiment(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            config=config,
            name=name,
        )

        results.append(
            result
        )

    print()

    print("=" * 60)

    print(
        "FINAL ABLATION SUMMARY"
    )

    print("=" * 60)

    save_experiment_results(
        results,
        "artifacts/ablation_results.json",
    )

    for result in results:

        print()

        print(
            result.name
        )

        print(
            f"Best Validation Accuracy: "
            f"{result.best_validation_accuracy:.4f}"
        )

        print(
            f"Training Time: "
            f"{result.training_time:.2f} seconds"
        )

        print(
            f"Epochs Recorded: "
            f"{len(result.epoch_results)}"
        )


if __name__ == "__main__":
    main()