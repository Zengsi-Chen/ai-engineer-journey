from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training_adapter import train_model


class FakeTracker:
    def __init__(self):
        self.record = {
            "experiment_id": "exp_integration_test",
            "training_history": [],
            "best_epoch": None,
            "best_metric": None,
            "best_checkpoint": None,
        }

    def log_epoch(self, metrics):
        self.record["training_history"].append(metrics)

        current_value = metrics["val_accuracy"]

        if (
            self.record["best_metric"] is None
            or current_value
            > self.record["best_metric"]["value"]
        ):
            self.record["best_epoch"] = metrics["epoch"]

            self.record["best_metric"] = {
                "name": "val_accuracy",
                "value": current_value,
            }

    def get_record(self):
        return self.record

    def set_best_checkpoint(self, path):
        self.record["best_checkpoint"] = {
            "path": path,
            "epoch": self.record["best_epoch"],
            "metric_name": self.record["best_metric"]["name"],
            "metric_value": self.record["best_metric"]["value"],
        }


class FakeConfig:
    class Training:
        epochs = 2
        learning_rate = 0.01
        device = "cpu"

    class Checkpoint:
        output_dir = "artifacts/test_checkpoints"

    training = Training()
    checkpoint = Checkpoint()


def create_dataloader():
    x = torch.randn(4, 1)
    y = torch.tensor([0, 1, 0, 1])

    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )


def create_model():
    return nn.Linear(1, 2)


def test_training_saves_best_checkpoint(
    monkeypatch,
    tmp_path,
):
    """
    Verify that training:

    1. logs validation metrics,
    2. identifies the best epoch,
    3. saves the best checkpoint,
    4. reloads the best checkpoint,
    5. evaluates the restored model on the test set.
    """

    def fake_train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
    ):
        return 0.5, 0.75

    evaluation_results = [
        {
            "loss": 0.50,
            "accuracy": 0.70,
        },
        {
            "loss": 0.40,
            "accuracy": 0.85,
        },
        {
            "loss": 0.30,
            "accuracy": 0.90,
        },
    ]

    evaluation_calls = []

    def fake_evaluate_model(
        model,
        dataloader,
        config,
    ):
        evaluation_calls.append(dataloader)

        call_index = len(evaluation_calls) - 1

        return evaluation_results[call_index]

    import src.training_adapter as training_adapter

    monkeypatch.setattr(
        training_adapter,
        "_load_day25_training_module",
        lambda: type(
            "FakeTrainingModule",
            (),
            {
                "train_one_epoch": staticmethod(
                    fake_train_one_epoch
                )
            },
        )(),
    )

    monkeypatch.setattr(
        training_adapter,
        "evaluate_model",
        fake_evaluate_model,
    )

    tracker = FakeTracker()

    config = FakeConfig()

    config.Checkpoint.output_dir = str(
        tmp_path
    )

    train_loader = create_dataloader()
    validation_loader = create_dataloader()
    test_loader = create_dataloader()

    model = create_model()

    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
    )

    record = tracker.get_record()

    # -------------------------------------------------
    # 1. Verify training history
    # -------------------------------------------------

    assert len(
        record["training_history"]
    ) == 2

    assert (
        record["training_history"][0]["epoch"]
        == 1
    )

    assert (
        record["training_history"][1]["epoch"]
        == 2
    )

    # -------------------------------------------------
    # 2. Verify best epoch
    # -------------------------------------------------

    assert record["best_epoch"] == 2

    assert (
        record["best_metric"]["name"]
        == "val_accuracy"
    )

    assert (
        record["best_metric"]["value"]
        == 0.85
    )

    # -------------------------------------------------
    # 3. Verify checkpoint metadata
    # -------------------------------------------------

    checkpoint = (
        record["best_checkpoint"]
    )

    assert checkpoint is not None

    assert (
        checkpoint["epoch"]
        == 2
    )

    assert (
        checkpoint["metric_name"]
        == "val_accuracy"
    )

    assert (
        checkpoint["metric_value"]
        == 0.85
    )

    # -------------------------------------------------
    # 4. Verify checkpoint file exists
    # -------------------------------------------------

    checkpoint_path = Path(
        checkpoint["path"]
    )

    assert checkpoint_path.exists()

    assert checkpoint_path.is_file()

    assert checkpoint_path.name == (
        "exp_integration_test_best.pth"
    )

    # -------------------------------------------------
    # 5. Verify evaluation calls
    #
    # Epoch 1 -> validation
    # Epoch 2 -> validation
    # End    -> test
    # -------------------------------------------------

    assert len(
        evaluation_calls
    ) == 3

    assert (
        evaluation_calls[0]
        is validation_loader
    )

    assert (
        evaluation_calls[1]
        is validation_loader
    )

    assert (
        evaluation_calls[2]
        is test_loader
    )

    # -------------------------------------------------
    # 6. Verify final test metrics
    # -------------------------------------------------

    assert result["test_metrics"] == {
        "loss": 0.30,
        "accuracy": 0.90,
    }

    # -------------------------------------------------
    # 7. Verify result exposes best checkpoint
    # -------------------------------------------------

    assert (
        result["best_checkpoint"]
        == checkpoint
    )

    assert (
        result["best_epoch"]
        == 2
    )

    assert (
        result["best_metric"]
        == record["best_metric"]
    )

