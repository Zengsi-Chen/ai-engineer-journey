import json

import pytest

from src.experiment_comparator import (
    ExperimentComparator,
)


def create_experiment_file(
    directory,
    experiment_id,
    metric_value=None,
    checkpoint_path=None,
):
    experiment = {
        "experiment_id": experiment_id,
        "best_metric": None,
        "best_checkpoint": None,
    }

    if metric_value is not None:
        experiment["best_metric"] = {
            "name": "val_accuracy",
            "value": metric_value,
        }

    if checkpoint_path is not None:
        experiment["best_checkpoint"] = {
            "path": str(checkpoint_path),
            "epoch": 1,
            "metric_name": "val_accuracy",
            "metric_value": metric_value,
        }

    output_path = (
        directory / f"{experiment_id}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
            indent=4,
        )

    return output_path


def test_create_comparator(tmp_path):
    comparator = ExperimentComparator(tmp_path)

    assert comparator.experiments_dir == tmp_path
    assert comparator.experiments == []


def test_load_all_experiments(tmp_path):
    create_experiment_file(
        tmp_path,
        "exp_001",
        0.85,
    )

    create_experiment_file(
        tmp_path,
        "exp_002",
        0.90,
    )

    create_experiment_file(
        tmp_path,
        "exp_003",
        0.88,
    )

    comparator = ExperimentComparator(tmp_path)

    experiments = comparator.load_all()

    assert len(experiments) == 3
    assert comparator.get_experiment_count() == 3


def test_load_all_ignores_non_json_files(tmp_path):
    create_experiment_file(
        tmp_path,
        "exp_001",
        0.85,
    )

    notes_file = tmp_path / "notes.txt"

    notes_file.write_text(
        "This is not an experiment.",
        encoding="utf-8",
    )

    comparator = ExperimentComparator(tmp_path)

    experiments = comparator.load_all()

    assert len(experiments) == 1


def test_load_all_requires_existing_directory(
    tmp_path,
):
    experiments_dir = (
        tmp_path / "missing_experiments"
    )

    comparator = ExperimentComparator(
        experiments_dir
    )

    with pytest.raises(
        FileNotFoundError,
        match="Experiments directory not found",
    ):
        comparator.load_all()


def test_load_all_does_not_duplicate_experiments(
    tmp_path,
):
    create_experiment_file(
        tmp_path,
        "exp_001",
        0.85,
    )

    create_experiment_file(
        tmp_path,
        "exp_002",
        0.90,
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()
    comparator.load_all()

    assert comparator.get_experiment_count() == 2


def test_get_best_experiment(tmp_path):
    create_experiment_file(
        tmp_path,
        "exp_001",
        0.85,
    )

    create_experiment_file(
        tmp_path,
        "exp_002",
        0.93,
    )

    create_experiment_file(
        tmp_path,
        "exp_003",
        0.88,
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    best = comparator.get_best_experiment()

    assert best["experiment_id"] == "exp_002"

    assert (
        best["best_metric"]["value"]
        == 0.93
    )


def test_get_best_experiment_requires_loaded_experiments(
    tmp_path,
):
    comparator = ExperimentComparator(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="No experiments loaded",
    ):
        comparator.get_best_experiment()


def test_get_best_experiment_requires_best_metric(
    tmp_path,
):
    create_experiment_file(
        tmp_path,
        "exp_001",
    )

    create_experiment_file(
        tmp_path,
        "exp_002",
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    with pytest.raises(
        RuntimeError,
        match="No experiments contain a best metric",
    ):
        comparator.get_best_experiment()


def test_get_best_experiment_ignores_incomplete_experiments(
    tmp_path,
):
    create_experiment_file(
        tmp_path,
        "exp_incomplete",
    )

    create_experiment_file(
        tmp_path,
        "exp_001",
        0.85,
    )

    create_experiment_file(
        tmp_path,
        "exp_002",
        0.91,
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    best = comparator.get_best_experiment()

    assert best["experiment_id"] == "exp_002"


def test_load_all_uses_deterministic_order(tmp_path):
    create_experiment_file(
        tmp_path,
        "exp_003",
        0.88,
    )

    create_experiment_file(
        tmp_path,
        "exp_001",
        0.85,
    )

    create_experiment_file(
        tmp_path,
        "exp_002",
        0.90,
    )

    comparator = ExperimentComparator(tmp_path)

    experiments = comparator.load_all()

    experiment_ids = [
        experiment["experiment_id"]
        for experiment in experiments
    ]

    assert experiment_ids == [
        "exp_001",
        "exp_002",
        "exp_003",
    ]


def test_select_best_model(tmp_path):
    checkpoint_1 = (
        tmp_path / "exp_001_best.pt"
    )

    checkpoint_2 = (
        tmp_path / "exp_002_best.pt"
    )

    checkpoint_1.touch()
    checkpoint_2.touch()

    create_experiment_file(
        tmp_path,
        "exp_001",
        metric_value=0.85,
        checkpoint_path=checkpoint_1,
    )

    create_experiment_file(
        tmp_path,
        "exp_002",
        metric_value=0.93,
        checkpoint_path=checkpoint_2,
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    best_model = (
        comparator.select_best_model()
    )

    assert (
        best_model["experiment_id"]
        == "exp_002"
    )

    assert (
        best_model["checkpoint_path"]
        == str(checkpoint_2)
    )

    assert best_model["epoch"] == 1

    assert (
        best_model["metric_name"]
        == "val_accuracy"
    )

    assert (
        best_model["metric_value"]
        == 0.93
    )


def test_select_best_model_requires_checkpoint(
    tmp_path,
):
    create_experiment_file(
        tmp_path,
        "exp_001",
        metric_value=0.95,
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    with pytest.raises(
        RuntimeError,
        match="Best experiment does not have a checkpoint",
    ):
        comparator.select_best_model()


def test_select_best_model_requires_existing_file(
    tmp_path,
):
    missing_checkpoint = (
        tmp_path / "missing_model.pt"
    )

    create_experiment_file(
        tmp_path,
        "exp_001",
        metric_value=0.95,
        checkpoint_path=missing_checkpoint,
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint file not found",
    ):
        comparator.select_best_model()


def test_select_best_model_requires_complete_checkpoint_metadata(
    tmp_path,
):
    checkpoint = tmp_path / "model.pt"

    checkpoint.touch()

    experiment = {
        "experiment_id": "exp_001",
        "best_metric": {
            "name": "val_accuracy",
            "value": 0.95,
        },
        "best_checkpoint": {
            "path": str(checkpoint),
        },
    }

    experiment_file = (
        tmp_path / "exp_001.json"
    )

    with open(
        experiment_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
        )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    with pytest.raises(
        ValueError,
        match="Best checkpoint is missing required fields",
    ):
        comparator.select_best_model()


def test_checkpoint_path_must_be_file(tmp_path):
    checkpoint_directory = (
        tmp_path / "checkpoint_directory"
    )

    checkpoint_directory.mkdir()

    create_experiment_file(
        tmp_path,
        "exp_001",
        metric_value=0.95,
        checkpoint_path=checkpoint_directory,
    )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    with pytest.raises(
        ValueError,
        match="Checkpoint path is not a file",
    ):
        comparator.select_best_model()


def test_checkpoint_metric_must_match_experiment_metric(
    tmp_path,
):
    checkpoint = tmp_path / "model.pt"

    checkpoint.touch()

    experiment = {
        "experiment_id": "exp_001",
        "best_metric": {
            "name": "val_accuracy",
            "value": 0.95,
        },
        "best_checkpoint": {
            "path": str(checkpoint),
            "epoch": 10,
            "metric_name": "val_accuracy",
            "metric_value": 0.90,
        },
    }

    experiment_file = (
        tmp_path / "exp_001.json"
    )

    with open(
        experiment_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
        )

    comparator = ExperimentComparator(tmp_path)

    comparator.load_all()

    with pytest.raises(
        ValueError,
        match="Checkpoint metric value does not match",
    ):
        comparator.select_best_model()
