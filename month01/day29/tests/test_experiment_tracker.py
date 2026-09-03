import json
import re
import pytest
import torch

from src.experiment_tracker import ExperimentTracker


def test_create_experiment():
    tracker = ExperimentTracker("exp_001")

    record = tracker.get_record()

    assert record["experiment_id"] == "exp_001"
    assert "created_at" in record["metadata"]
    assert "python_version" in record["metadata"]
    assert "torch_version" in record["metadata"]

    assert record["hyperparameters"] == {}
    assert record["training_history"] == []

    assert record["best_checkpoint"] is None
    assert record["best_epoch"] is None
    assert record["best_metric"] is None


def create_epoch_metrics(epoch, val_accuracy):
    return {
        "epoch": epoch,
        "train_loss": 1.0 - val_accuracy,
        "val_loss": 1.0 - val_accuracy,
        "train_accuracy": val_accuracy,
        "val_accuracy": val_accuracy,
    }


def test_first_epoch_becomes_best():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(
            epoch=1,
            val_accuracy=0.72,
        )
    )

    record = tracker.get_record()

    assert record["best_epoch"] == 1
    assert record["best_metric"] == {
        "name": "val_accuracy",
        "value": 0.72,
    }


def test_higher_accuracy_updates_best_metric():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(1, 0.72)
    )

    tracker.log_epoch(
        create_epoch_metrics(2, 0.81)
    )

    record = tracker.get_record()

    assert record["best_epoch"] == 2
    assert record["best_metric"]["value"] == 0.81


def test_lower_accuracy_does_not_update_best_metric():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(1, 0.80)
    )

    tracker.log_epoch(
        create_epoch_metrics(2, 0.75)
    )

    record = tracker.get_record()

    assert record["best_epoch"] == 1
    assert record["best_metric"]["value"] == 0.80


def test_equal_accuracy_keeps_earlier_best_epoch():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(1, 0.85)
    )

    tracker.log_epoch(
        create_epoch_metrics(2, 0.85)
    )

    record = tracker.get_record()

    assert record["best_epoch"] == 1
    assert record["best_metric"]["value"] == 0.85


def test_best_metric_tracking_across_multiple_epochs():
    tracker = ExperimentTracker("exp_001")

    accuracies = [
        0.72,
        0.81,
        0.87,
        0.85,
        0.86,
    ]

    for epoch, accuracy in enumerate(accuracies, start=1):
        tracker.log_epoch(
            create_epoch_metrics(
                epoch=epoch,
                val_accuracy=accuracy,
            )
        )

    record = tracker.get_record()

    assert record["best_epoch"] == 3
    assert record["best_metric"] == {
        "name": "val_accuracy",
        "value": 0.87,
    }


def test_log_hyperparameters():
    tracker = ExperimentTracker("exp_001")

    params = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
    }

    tracker.log_hyperparameters(params)

    record = tracker.get_record()

    assert record["hyperparameters"] == params


def test_hyperparameters_merge():
    tracker = ExperimentTracker("exp_001")

    tracker.log_hyperparameters({
        "learning_rate": 0.001
    })

    tracker.log_hyperparameters({
        "batch_size": 64
    })

    tracker.log_hyperparameters({
        "epochs": 20
    })

    record = tracker.get_record()

    assert record["hyperparameters"] == {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
    }


def test_invalid_learning_rate():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(
        ValueError,
        match="learning_rate must be greater than 0"
    ):
        tracker.log_hyperparameters({
            "learning_rate": 0
        })


def test_negative_learning_rate():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(ValueError):
        tracker.log_hyperparameters({
            "learning_rate": -0.001
        })


def test_invalid_batch_size():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(
        ValueError,
        match="batch_size must be a positive integer"
    ):
        tracker.log_hyperparameters({
            "batch_size": 0
        })


def test_batch_size_must_be_integer():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(ValueError):
        tracker.log_hyperparameters({
            "batch_size": 32.0
        })


def test_batch_size_cannot_be_bool():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(ValueError):
        tracker.log_hyperparameters({
            "batch_size": True
        })


def test_invalid_epochs():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(
        ValueError,
        match="epochs must be a positive integer"
    ):
        tracker.log_hyperparameters({
            "epochs": 0
        })


def test_epochs_must_be_integer():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(ValueError):
        tracker.log_hyperparameters({
            "epochs": 20.0
        })


def test_invalid_hyperparameters_do_not_modify_record():
    tracker = ExperimentTracker("exp_001")

    tracker.log_hyperparameters({
        "learning_rate": 0.001
    })

    with pytest.raises(ValueError):
        tracker.log_hyperparameters({
            "batch_size": 0
        })

    record = tracker.get_record()

    assert record["hyperparameters"] == {
        "learning_rate": 0.001
    }


def test_log_metadata():
    tracker = ExperimentTracker("exp_001")

    metadata = {
        "dataset": "MNIST",
        "model": "CNN",
        "device": "cpu",
    }

    tracker.log_metadata(metadata)

    record = tracker.get_record()

    assert record["metadata"]["dataset"] == "MNIST"
    assert record["metadata"]["model"] == "CNN"
    assert record["metadata"]["device"] == "cpu"

    assert "created_at" in record["metadata"]
    assert "python_version" in record["metadata"]
    assert "torch_version" in record["metadata"]


def test_metadata_merge():
    tracker = ExperimentTracker("exp_001")

    tracker.log_metadata({
        "dataset": "MNIST"
    })

    tracker.log_metadata({
        "device": "cpu"
    })

    record = tracker.get_record()

    assert record["metadata"]["dataset"] == "MNIST"
    assert record["metadata"]["device"] == "cpu"


def test_automatic_metadata():
    tracker = ExperimentTracker("exp_001")

    metadata = tracker.get_record()["metadata"]

    assert metadata["created_at"]
    assert metadata["python_version"]
    assert metadata["torch_version"]


def test_log_epoch():
    tracker = ExperimentTracker("exp_001")

    metrics = {
        "epoch": 1,
        "train_loss": 0.82,
        "val_loss": 0.71,
        "train_accuracy": 0.70,
        "val_accuracy": 0.72,
    }

    tracker.log_epoch(metrics)

    record = tracker.get_record()

    assert len(record["training_history"]) == 1
    assert record["training_history"][0] == metrics


def test_log_multiple_epochs():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch({
        "epoch": 1,
        "train_loss": 0.8,
        "val_loss": 0.7,
        "train_accuracy": 0.70,
        "val_accuracy": 0.72,
    })

    tracker.log_epoch({
        "epoch": 2,
        "train_loss": 0.6,
        "val_loss": 0.5,
        "train_accuracy": 0.80,
        "val_accuracy": 0.82,
    })

    history = tracker.get_record()["training_history"]

    assert len(history) == 2
    assert history[1]["epoch"] == 2


def test_missing_required_metrics():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(
        ValueError,
        match="Missing required metrics"
    ):
        tracker.log_epoch({
            "epoch": 1,
            "train_loss": 0.8,
        })


def test_first_epoch_must_be_one():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(
        ValueError,
        match="Expected epoch 1"
    ):
        tracker.log_epoch({
            "epoch": 2,
            "train_loss": 0.8,
            "val_loss": 0.7,
            "train_accuracy": 0.70,
            "val_accuracy": 0.72,
        })


def test_epoch_cannot_skip():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch({
        "epoch": 1,
        "train_loss": 0.8,
        "val_loss": 0.7,
        "train_accuracy": 0.70,
        "val_accuracy": 0.72,
    })

    with pytest.raises(
        ValueError,
        match="Expected epoch 2"
    ):
        tracker.log_epoch({
            "epoch": 3,
            "train_loss": 0.6,
            "val_loss": 0.5,
            "train_accuracy": 0.80,
            "val_accuracy": 0.82,
        })


def test_epoch_cannot_repeat():
    tracker = ExperimentTracker("exp_001")

    metrics = {
        "epoch": 1,
        "train_loss": 0.8,
        "val_loss": 0.7,
        "train_accuracy": 0.70,
        "val_accuracy": 0.72,
    }

    tracker.log_epoch(metrics)

    with pytest.raises(
        ValueError,
        match="Expected epoch 2"
    ):
        tracker.log_epoch(metrics)


def test_invalid_epoch_does_not_modify_history():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch({
        "epoch": 1,
        "train_loss": 0.8,
        "val_loss": 0.7,
        "train_accuracy": 0.70,
        "val_accuracy": 0.72,
    })

    with pytest.raises(ValueError):
        tracker.log_epoch({
            "epoch": 3,
            "train_loss": 0.6,
            "val_loss": 0.5,
            "train_accuracy": 0.80,
            "val_accuracy": 0.82,
        })

    history = tracker.get_record()["training_history"]

    assert len(history) == 1
    assert history[0]["epoch"] == 1


def test_set_best_checkpoint():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(
            epoch=1,
            val_accuracy=0.87,
        )
    )

    checkpoint_path = (
        "artifacts/checkpoints/exp_001_best.pt"
    )

    tracker.set_best_checkpoint(checkpoint_path)

    record = tracker.get_record()

    assert record["best_checkpoint"] == {
        "path": checkpoint_path,
        "epoch": 1,
        "metric_name": "val_accuracy",
        "metric_value": 0.87,
    }


def test_checkpoint_requires_best_epoch():
    tracker = ExperimentTracker("exp_001")

    with pytest.raises(
        RuntimeError,
        match="Cannot set best checkpoint"
    ):
        tracker.set_best_checkpoint(
            "artifacts/checkpoints/model.pt"
        )


def test_checkpoint_path_must_be_non_empty():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(
            epoch=1,
            val_accuracy=0.87,
        )
    )

    with pytest.raises(
        ValueError,
        match="Checkpoint path must be a non-empty string"
    ):
        tracker.set_best_checkpoint("")


def test_checkpoint_path_must_be_string():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(
            epoch=1,
            val_accuracy=0.87,
        )
    )

    with pytest.raises(ValueError):
        tracker.set_best_checkpoint(None)


def test_checkpoint_tracks_current_best_metric():
    tracker = ExperimentTracker("exp_001")

    tracker.log_epoch(
        create_epoch_metrics(
            epoch=1,
            val_accuracy=0.80,
        )
    )

    tracker.log_epoch(
        create_epoch_metrics(
            epoch=2,
            val_accuracy=0.90,
        )
    )

    checkpoint_path = (
        "artifacts/checkpoints/exp_001_best.pt"
    )

    tracker.set_best_checkpoint(checkpoint_path)

    checkpoint = (
        tracker.get_record()["best_checkpoint"]
    )

    assert checkpoint["path"] == checkpoint_path
    assert checkpoint["epoch"] == 2
    assert checkpoint["metric_name"] == "val_accuracy"
    assert checkpoint["metric_value"] == 0.90


def test_generate_experiment_id():
    tracker = ExperimentTracker()

    record = tracker.get_record()
    experiment_id = record["experiment_id"]

    assert experiment_id.startswith("exp_")


def test_experiment_ids_are_unique():
    tracker1 = ExperimentTracker()
    tracker2 = ExperimentTracker()

    id1 = tracker1.get_record()["experiment_id"]
    id2 = tracker2.get_record()["experiment_id"]

    assert id1 != id2


def test_manual_experiment_id():
    tracker = ExperimentTracker("exp_manual")

    record = tracker.get_record()

    assert record["experiment_id"] == "exp_manual"


def test_experiment_id_format():
    tracker = ExperimentTracker()

    experiment_id = tracker.get_record()["experiment_id"]

    pattern = r"^exp_\d{8}_\d{6}_[a-f0-9]{8}$"

    assert re.match(pattern, experiment_id)


def test_save_experiment_record(tmp_path):
    tracker = ExperimentTracker("exp_001")

    output_path = tracker.save(tmp_path)

    assert output_path.exists()
    assert output_path.name == "exp_001.json"


def test_saved_json_content(tmp_path):
    tracker = ExperimentTracker("exp_001")

    tracker.log_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
    })

    output_path = tracker.save(tmp_path)

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as file:
        saved_record = json.load(file)

    assert saved_record["experiment_id"] == "exp_001"

    assert (
        saved_record["hyperparameters"]["learning_rate"]
        == 0.001
    )

    assert (
        saved_record["hyperparameters"]["batch_size"]
        == 64
    )

    assert (
        saved_record["hyperparameters"]["epochs"]
        == 20
    )


def test_load_experiment_record(tmp_path):
    tracker = ExperimentTracker("exp_001")

    tracker.log_metadata({
        "dataset": "MNIST"
    })

    output_path = tracker.save(tmp_path)

    loaded_record = ExperimentTracker.load(
        output_path
    )

    assert (
        loaded_record["experiment_id"]
        == "exp_001"
    )

    assert (
        loaded_record["metadata"]["dataset"]
        == "MNIST"
    )


def test_save_and_load_round_trip(tmp_path):
    tracker = ExperimentTracker("exp_001")

    tracker.log_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
    })

    tracker.log_epoch(
        create_epoch_metrics(
            epoch=1,
            val_accuracy=0.80,
        )
    )

    tracker.set_best_checkpoint(
        "artifacts/checkpoints/exp_001_best.pt"
    )

    original_record = tracker.get_record()

    output_path = tracker.save(tmp_path)

    loaded_record = ExperimentTracker.load(
        output_path
    )

    assert loaded_record == original_record


def test_tracker_initializes_reproducibility_as_none():
    tracker = ExperimentTracker("exp_001")

    record = tracker.get_record()

    assert record["reproducibility"] is None


def test_set_reproducibility():
    tracker = ExperimentTracker("exp_001")

    tracker.set_reproducibility(42)

    record = tracker.get_record()

    reproducibility = record["reproducibility"]

    assert reproducibility["seed"] == 42

    assert "environment" in reproducibility

    assert (
        reproducibility["environment"]["device"]
        in {"cpu", "cuda"}
    )


def test_tracker_reproducibility_sets_seed():
    tracker = ExperimentTracker("exp_001")

    tracker.set_reproducibility(42)

    first_value = torch.rand(5)

    tracker.set_reproducibility(42)

    second_value = torch.rand(5)

    assert torch.equal(
        first_value,
        second_value,
    )