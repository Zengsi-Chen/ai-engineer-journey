import json
from pathlib import Path


def test_capstone_artifacts_are_consistent():
    experiment_dir = Path(
        "artifacts/experiments"
    )
    checkpoint_dir = Path(
        "artifacts/checkpoints"
    )

    experiment_files = list(
        experiment_dir.glob("*.json")
    )

    assert experiment_files, (
        "No experiment JSON file was found."
    )

    experiment_path = max(
        experiment_files,
        key=lambda path: path.stat().st_mtime,
    )

    with open(
        experiment_path,
        "r",
        encoding="utf-8",
    ) as file:
        record = json.load(file)

    required_fields = {
        "experiment_id",
        "metadata",
        "hyperparameters",
        "training_history",
        "best_epoch",
        "best_metric",
        "best_checkpoint",
        "reproducibility",
    }

    assert required_fields.issubset(
        record.keys()
    )

    assert len(
        record["training_history"]
    ) == record["hyperparameters"]["epochs"]

    assert record["best_epoch"] is not None

    best_metric = record["best_metric"]

    assert best_metric["name"] == "val_accuracy"

    best_checkpoint = record[
        "best_checkpoint"
    ]

    assert best_checkpoint is not None

    checkpoint_path = Path(
        best_checkpoint["path"]
    )

    assert checkpoint_path.exists(), (
        f"Checkpoint does not exist: "
        f"{checkpoint_path}"
    )

    assert (
        best_checkpoint["epoch"]
        == record["best_epoch"]
    )

    assert (
        best_checkpoint["metric_name"]
        == best_metric["name"]
    )

    assert (
        best_checkpoint["metric_value"]
        == best_metric["value"]
    )

    assert (
        best_checkpoint["path"]
        == str(checkpoint_path)
    )

