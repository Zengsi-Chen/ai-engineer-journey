from dataclasses import dataclass


@dataclass
class StrategyResult:
    name: str
    best_accuracy: float
    best_epoch: int
    training_time: float


def calculate_accuracy_gain(
    baseline_accuracy,
    fine_tuned_accuracy,
):
    return (
        fine_tuned_accuracy
        - baseline_accuracy
    )


def compare_strategies(
    baseline_accuracy,
    results,
):
    comparison = []

    for result in results:

        accuracy_gain = (
            calculate_accuracy_gain(
                baseline_accuracy=baseline_accuracy,
                fine_tuned_accuracy=(
                    result.best_accuracy
                ),
            )
        )

        accuracy_per_second = (
            calculate_accuracy_per_second(
                accuracy_gain=accuracy_gain,
                training_time=result.training_time,
            )
        )

        comparison.append(
            {
                "strategy": result.name,
                "best_accuracy": (
                    result.best_accuracy
                ),
                "best_epoch": (
                    result.best_epoch
                ),
                "training_time": (
                    result.training_time
                ),
                "accuracy_gain": (
                    accuracy_gain
                ),
                "accuracy_per_second": (
                    accuracy_per_second
                ),
            }
        )        

    return comparison


def find_best_strategy(
    results,
):
    return max(
        results,
        key=lambda result: (
            result.best_accuracy
        ),
    )


def calculate_time_increase(
    baseline_time,
    training_time,
):
    return (
        training_time
        - baseline_time
    )


def calculate_accuracy_per_second(
    accuracy_gain,
    training_time,
):
    if training_time <= 0:
        return 0.0

    return (
        accuracy_gain
        / training_time
    )

def find_most_efficient_strategy(
    comparison,
):
    return max(
        comparison,
        key=lambda item: (
            item["accuracy_per_second"]
        ),
    )