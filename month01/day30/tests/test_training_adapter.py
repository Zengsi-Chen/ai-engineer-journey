from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training_adapter import train_model


class FakeTracker:
    def __init__(self):
        self.record = {
            "experiment_id": "exp_test",
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


def test_train_model_logs_every_epoch(monkeypatch):
    """
    Training should log exactly one metric record per epoch.
    """

    def fake_train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
    ):
        return 0.5, 0.75

    def fake_evaluate_model(
        model,
        dataloader,
        config,
    ):
        return {
            "loss": 0.4,
            "accuracy": 0.8,
        }

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

    def fake_save_checkpoint(model, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            model.state_dict(),
            path,
        )

        return path

    tracker = FakeTracker()

    config = FakeConfig()

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

    history = tracker.get_record()["training_history"]

    assert len(history) == config.training.epochs

    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2

    assert result["best_epoch"] == 1

    assert result["best_metric"]["name"] == "val_accuracy"
    assert result["best_metric"]["value"] == 0.8


def test_train_model_saves_best_checkpoint(monkeypatch, tmp_path):
    """
    Training should save a checkpoint when a new best
    validation metric is found.
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
            "loss": 0.5,
            "accuracy": 0.70,
        },
        {
            "loss": 0.4,
            "accuracy": 0.85,
        },
        {
            "loss": 0.3,
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

        return evaluation_results[
            len(evaluation_calls) - 1
        ]

    saved_paths = []

    def fake_save_checkpoint(
        model,
        path,
    ):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            path,
        )

        saved_paths.append(path)

        return path

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

    monkeypatch.setattr(
        training_adapter,
        "save_checkpoint",
        fake_save_checkpoint,
    )

    tracker = FakeTracker()

    config = FakeConfig()
    config.Checkpoint.output_dir = str(tmp_path)

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

    checkpoint = tracker.get_record()["best_checkpoint"]

    assert checkpoint is not None

    assert checkpoint["epoch"] == 2

    assert checkpoint["metric_name"] == "val_accuracy"

    assert checkpoint["metric_value"] == 0.85

    assert len(saved_paths) == 2

    assert saved_paths[0].name == "exp_test_best.pth"

    assert saved_paths[1].name == "exp_test_best.pth"

    assert result["best_epoch"] == 2

    assert result["test_metrics"]["accuracy"] == 0.90


def test_train_model_propagates_training_error(monkeypatch):
    """
    Training errors should not be swallowed by the adapter.
    """

    def fake_train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
    ):
        raise RuntimeError(
            "training failed"
        )

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

    tracker = FakeTracker()

    config = FakeConfig()

    train_loader = create_dataloader()
    validation_loader = create_dataloader()
    test_loader = create_dataloader()

    model = create_model()

    try:
        train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            config=config,
            tracker=tracker,
        )
    except RuntimeError as error:
        assert str(error) == "training failed"
    else:
        raise AssertionError(
            "RuntimeError was not propagated."
        )

