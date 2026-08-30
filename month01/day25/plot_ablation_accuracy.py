import matplotlib.pyplot as plt

from experiment_results import (
    load_experiment_results,
)


def plot_accuracy_curves(
    results,
):

    plot_training_accuracy(
        results
    )

    plot_validation_accuracy(
        results
    )


def plot_training_accuracy(
    results,
):

    plt.figure(
        figsize=(10, 6)
    )

    for result in results:

        epochs = [
            epoch_result.epoch
            for epoch_result
            in result.epoch_results
        ]

        train_accuracies = [
            epoch_result.train_accuracy
            for epoch_result
            in result.epoch_results
        ]

        plt.plot(
            epochs,
            train_accuracies,
            marker="o",
            label=result.name,
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Accuracy"
    )

    plt.title(
        "ResNet Ablation: Training Accuracy"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "artifacts/ablation_training_accuracy.png"
    )

    plt.show()


def plot_validation_accuracy(
    results,
):

    plt.figure(
        figsize=(10, 6)
    )

    for result in results:

        epochs = [
            epoch_result.epoch
            for epoch_result
            in result.epoch_results
        ]

        validation_accuracies = [
            epoch_result.validation_accuracy
            for epoch_result
            in result.epoch_results
        ]

        plt.plot(
            epochs,
            validation_accuracies,
            marker="o",
            label=result.name,
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation Accuracy"
    )

    plt.title(
        "ResNet Ablation: Validation Accuracy"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "artifacts/ablation_validation_accuracy.png"
    )

    plt.show()


def main():

    results = load_experiment_results(
        "artifacts/ablation_results.json"
    )

    plot_accuracy_curves(
        results
    )


if __name__ == "__main__":
    main()