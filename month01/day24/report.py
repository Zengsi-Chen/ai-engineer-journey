import json
from pathlib import Path


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_report(data):
    config = data["config"]
    results = data["results"]

    print("\n" + "=" * 70)
    print("TRANSFER LEARNING EXPERIMENT REPORT")
    print("=" * 70)

    print("\nExperiment Configuration")
    print("-" * 70)

    print(f"Seed:             {config['seed']}")
    print(f"Batch Size:       {config['batch_size']}")
    print(f"Epochs:            {config['epochs']}")
    print(f"Classifier LR:     {config['classifier_lr']}")
    print(f"Backbone LR:       {config['backbone_lr']}")
    print(f"Classes:           {config['num_classes']}")

    print("\nResults")
    print("-" * 70)

    print(
        f"{'Experiment':<22}"
        f"{'Params':>12}"
        f"{'Best Acc':>12}"
        f"{'Best Epoch':>14}"
        f"{'Time(s)':>12}"
    )

    print("-" * 70)

    for name, result in results.items():
        print(
            f"{name:<22}"
            f"{result['trainable_parameters']:>12,}"
            f"{result['best_test_accuracy']:>12.4f}"
            f"{result['best_epoch']:>14}"
            f"{result['training_time']:>12.2f}"
        )

    print("-" * 70)


def print_analysis(data):
    results = data["results"]

    feature = results["Feature Extraction"]
    fine_tuning = results["Fine-Tuning"]

    accuracy_gain = (
        fine_tuning["best_test_accuracy"]
        - feature["best_test_accuracy"]
    )

    parameter_ratio = (
        fine_tuning["trainable_parameters"]
        / feature["trainable_parameters"]
    )

    time_ratio = (
        fine_tuning["training_time"]
        / feature["training_time"]
    )

    print("\nAnalysis")
    print("-" * 70)

    print(
        f"Accuracy Gain: "
        f"{accuracy_gain:+.4f}"
    )

    print(
        f"Parameter Ratio: "
        f"{parameter_ratio:.1f}x"
    )

    print(
        f"Training Time Ratio: "
        f"{time_ratio:.2f}x"
    )


def print_recommendation(data):
    results = data["results"]

    feature = results["Feature Extraction"]
    fine_tuning = results["Fine-Tuning"]

    accuracy_gain = (
        fine_tuning["best_test_accuracy"]
        - feature["best_test_accuracy"]
    )

    if accuracy_gain > 0.01:
        recommendation = (
            "Fine-Tuning provides a meaningful "
            "accuracy improvement."
        )
    else:
        recommendation = (
            "Feature Extraction may provide a "
            "better efficiency/performance trade-off."
        )

    print("\nRecommendation")
    print("-" * 70)
    print(recommendation)


def main():
    results_path = Path(
        "artifacts/transfer_learning_results.json"
    )

    data = load_results(results_path)

    print_report(data)
    print_analysis(data)
    print_recommendation(data)


if __name__ == "__main__":
    main()