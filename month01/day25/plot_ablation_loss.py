import matplotlib.pyplot as plt

from experiment_results import (
    load_experiment_results,
)


def plot_loss_curves(
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

        validation_losses = [
            epoch_result.validation_loss
            for epoch_result
            in result.epoch_results
        ]

        plt.plot(
            epochs,
            validation_losses,
            marker="o",
            label=result.name,
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation Loss"
    )

    plt.title(
        "ResNet Ablation: Validation Loss"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "artifacts/ablation_validation_loss.png"
    )

    plt.show()


def main():

    results = load_experiment_results(
        "artifacts/ablation_results.json"
    )

    plot_loss_curves(
        results
    )


if __name__ == "__main__":
    main()