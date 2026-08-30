from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class EpochResult:

    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass
class ExperimentResult:

    name: str
    epoch_results: list[EpochResult]
    best_validation_accuracy: float
    training_time: float


def save_experiment_results(
    results,
    file_path,
):

    data = [
        asdict(result)
        for result in results
    ]

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def load_experiment_results(
    file_path,
):

    file_path = Path(file_path)

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    results = []

    for experiment_data in data:

        epoch_results = [
            EpochResult(
                **epoch_data
            )
            for epoch_data
            in experiment_data[
                "epoch_results"
            ]
        ]

        result = ExperimentResult(
            name=experiment_data["name"],
            epoch_results=epoch_results,
            best_validation_accuracy=(
                experiment_data[
                    "best_validation_accuracy"
                ]
            ),
            training_time=(
                experiment_data[
                    "training_time"
                ]
            ),
        )

        results.append(result)

    return results