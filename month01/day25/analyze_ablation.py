from experiment_results import (
    load_experiment_results,
)


def analyze_result(result):

    final_epoch = result.epoch_results[-1]

    generalization_gap = (
        final_epoch.train_accuracy
        - final_epoch.validation_accuracy
    )

    return {
        "model": result.name,
        "best_validation_accuracy": (
            result.best_validation_accuracy
        ),
        "final_train_accuracy": (
            final_epoch.train_accuracy
        ),
        "final_validation_accuracy": (
            final_epoch.validation_accuracy
        ),
        "final_validation_loss": (
            final_epoch.validation_loss
        ),
        "generalization_gap": (
            generalization_gap
        ),
        "training_time": (
            result.training_time
        ),
    }


def main():

    results = load_experiment_results(
        "artifacts/ablation_results.json"
    )

    analyses = [
        analyze_result(result)
        for result in results
    ]

    print()
    print("=" * 80)
    print("RESNET ABLATION ANALYSIS")
    print("=" * 80)

    for analysis in analyses:

        print()
        print(
            f"Model: "
            f"{analysis['model']}"
        )

        print(
            f"Best Val Accuracy: "
            f"{analysis['best_validation_accuracy']:.4f}"
        )

        print(
            f"Final Train Accuracy: "
            f"{analysis['final_train_accuracy']:.4f}"
        )

        print(
            f"Final Val Accuracy: "
            f"{analysis['final_validation_accuracy']:.4f}"
        )

        print(
            f"Final Val Loss: "
            f"{analysis['final_validation_loss']:.4f}"
        )

        print(
            f"Generalization Gap: "
            f"{analysis['generalization_gap']:.4f}"
        )

        print(
            f"Training Time: "
            f"{analysis['training_time']:.2f}s"
        )

    best_result = max(
        analyses,
        key=lambda x:
        x["best_validation_accuracy"],
    )

    print()
    print("=" * 80)

    print(
        "BEST MODEL"
    )

    print("=" * 80)

    print(
        best_result["model"]
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_result['best_validation_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()