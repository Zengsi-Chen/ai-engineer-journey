import json
from dataclasses import dataclass

from experiment_report import (
    generate_experiment_report,
)

RESULTS_PATH = (
    "artifacts/results/day27_results.json"
)

REPORT_PATH = (
    "artifacts/reports/"
    "day27_experiment_report.md"
)

@dataclass
class ExperimentResult:


    strategy_name: str
    best_epoch: int
    best_accuracy: float
    final_accuracy: float
    training_time: float


def load_results():

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        raw_results = json.load(
            file
        )

    results = {}

    for name, data in raw_results.items():

        result = ExperimentResult(
            strategy_name=(
                data["strategy_name"]
            ),
            best_epoch=(
                data["best_epoch"]
            ),
            best_accuracy=(
                data["best_accuracy"]
            ),
            final_accuracy=(
                data["final_accuracy"]
            ),
            training_time=(
                data["training_time"]
            ),
        )

        results[name] = {
            "result": result
        }

    return results


def main():


    results = load_results()

    generate_experiment_report(
        results=results,
        report_path=REPORT_PATH,
    )


if __name__ == "__main__":
    main()
