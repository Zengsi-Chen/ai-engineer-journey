from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training_adapter import train_model


class FakeTracker:
    def __init__(self):
        self.record = {
            "experiment_id": "exp_final_eval_test",
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


def test_training_evaluates_best_model_on_test_set(
    monkeypatch,
    tmp_path,
):
    """
    Verify the complete final evaluation flow:

    1. Train for multiple epochs.
    2. Evaluate on validation data after each epoch.
    3. Identify the best epoch.
    4. Save the best checkpoint.
    5. Reload the best checkpoint.
    6. Evaluate the restored model on the test set.
    7. Return the final test metrics.
    """

    # -------------------------------------------------
    # Fake training implementation
    # -------------------------------------------------

    def fake_train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
    ):
        return 0.50, 0.75

    # -------------------------------------------------
    # Evaluation results
    #
    # Call 1 -> Epoch 1 validation
    # Call 2 -> Epoch 2 validation
    # Call 3 -> Final test evaluation
    # -------------------------------------------------

    evaluation_results = [
        {
            "loss": 0.60,
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
        evaluation_calls.append(
            dataloader
        )

        call_index = (
            len(evaluation_calls) - 1
        )

        return evaluation_results[
            call_index
        ]

    # -------------------------------------------------
    # Import module so dependencies can be patched
    # -------------------------------------------------

    import src.training_adapter as training_adapter

    # -------------------------------------------------
    # Mock Day 25 training function
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Mock evaluation adapter
    #
    # This prevents the test from requiring CIFAR-10.
    # -------------------------------------------------

    monkeypatch.setattr(
        training_adapter,
        "evaluate_model",
        fake_evaluate_model,
    )

    # -------------------------------------------------
    # Use the REAL save_checkpoint implementation.
    #
    # This means the integration test verifies that
    # a real checkpoint file is actually created.
    # -------------------------------------------------

    tracker = FakeTracker()

    config = FakeConfig()

    config.Checkpoint.output_dir = str(
        tmp_path
    )

    train_loader = create_dataloader()

    validation_loader = create_dataloader()

    test_loader = create_dataloader()

    model = create_model()

    # -------------------------------------------------
    # Run training pipeline
    # -------------------------------------------------

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
    # 1. Verify validation happened every epoch
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

    # -------------------------------------------------
    # 2. Verify the final evaluation used TEST loader
    # -------------------------------------------------

    assert (
        evaluation_calls[2]
        is test_loader
    )

    # -------------------------------------------------
    # 3. Verify best epoch
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
    # 4. Verify best checkpoint exists
    # -------------------------------------------------

    best_checkpoint = (
        record["best_checkpoint"]
    )

    assert best_checkpoint is not None

    checkpoint_path = Path(
        best_checkpoint["path"]
    )

    assert checkpoint_path.exists()

    assert checkpoint_path.is_file()

    assert checkpoint_path.name == (
        "exp_final_eval_test_best.pth"
    )

    # -------------------------------------------------
    # 5. Verify final test metrics
    # -------------------------------------------------

    assert result["test_metrics"] == {
        "loss": 0.30,
        "accuracy": 0.90,
    }

    # -------------------------------------------------
    # 6. Verify result exposes best-model information
    # -------------------------------------------------

    assert (
        result["best_epoch"]
        == record["best_epoch"]
    )

    assert (
        result["best_metric"]
        == record["best_metric"]
    )

    assert (
        result["best_checkpoint"]
        == record["best_checkpoint"]
    )
