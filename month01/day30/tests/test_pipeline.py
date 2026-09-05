from pathlib import Path
from unittest.mock import Mock

from configs.pipeline_config import PipelineConfig
from src.pipeline import run_pipeline


class FakeExperimentAdapter:
    """
    Fake ExperimentAdapter used to verify that
    the Pipeline initializes the experiment before
    training starts.
    """

    initialized = False

    def __init__(
        self,
        config,
        tracker,
    ):
        self.config = config
        self.tracker = tracker

    def initialize(self):
        FakeExperimentAdapter.initialized = True

        return self.tracker


class FakeTracker:
    def __init__(self):
        self.record = {
            "experiment_id": "exp_pipeline_test",
        }

        self.save_called = False

    def save(self):
        self.save_called = True

        return Path(
            "artifacts/experiments/"
            "exp_pipeline_test.json"
        )

    def get_record(self):
        return self.record


def test_run_pipeline_orchestrates_all_components(
    monkeypatch,
):
    """
    Verify that run_pipeline correctly connects:

    ExperimentAdapter
        ↓
    Data Adapter
        ↓
    Model Adapter
        ↓
    Training Adapter
        ↓
    Tracker.save()

    No real dataset or model training should occur.
    """

    config = PipelineConfig()

    tracker = FakeTracker()

    fake_train_loader = Mock(
        name="train_loader"
    )

    fake_validation_loader = Mock(
        name="validation_loader"
    )

    fake_test_loader = Mock(
        name="test_loader"
    )

    fake_model = Mock(
        name="model"
    )

    fake_training_result = {
        "best_epoch": 2,
        "best_metric": {
            "name": "val_accuracy",
            "value": 0.85,
        },
        "best_checkpoint": {
            "path": (
                "artifacts/checkpoints/"
                "exp_pipeline_test_best.pth"
            ),
            "epoch": 2,
            "metric_name": "val_accuracy",
            "metric_value": 0.85,
        },
        "test_metrics": {
            "loss": 0.30,
            "accuracy": 0.90,
        },
    }

    # -------------------------------------------------
    # Track calls
    # -------------------------------------------------

    calls = []

    # -------------------------------------------------
    # Fake ExperimentAdapter
    # -------------------------------------------------

    monkeypatch.setattr(
        "src.pipeline.ExperimentAdapter",
        FakeExperimentAdapter,
    )

    # -------------------------------------------------
    # Fake Data Adapter
    # -------------------------------------------------

    def fake_create_data_loaders(
        received_config,
    ):
        calls.append(
            (
                "create_data_loaders",
                received_config,
            )
        )

        assert (
            received_config is config
        )

        return (
            fake_train_loader,
            fake_validation_loader,
            fake_test_loader,
        )

    monkeypatch.setattr(
        "src.pipeline.create_data_loaders",
        fake_create_data_loaders,
    )

    # -------------------------------------------------
    # Fake Model Adapter
    # -------------------------------------------------

    def fake_create_model(
        received_config,
    ):
        calls.append(
            (
                "create_model",
                received_config,
            )
        )

        assert (
            received_config is config
        )

        return fake_model

    monkeypatch.setattr(
        "src.pipeline.create_model",
        fake_create_model,
    )

    # -------------------------------------------------
    # Fake Training Adapter
    # -------------------------------------------------

    def fake_train_model(
        model,
        train_loader,
        validation_loader,
        test_loader,
        config,
        tracker,
    ):
        calls.append(
            (
                "train_model",
                model,
                train_loader,
                validation_loader,
                test_loader,
                config,
                tracker,
            )
        )

        assert model is fake_model

        assert (
            train_loader
            is fake_train_loader
        )

        assert (
            validation_loader
            is fake_validation_loader
        )

        assert (
            test_loader
            is fake_test_loader
        )

        assert config is config

        assert tracker is tracker

        return fake_training_result

    monkeypatch.setattr(
        "src.pipeline.train_model",
        fake_train_model,
    )

    # -------------------------------------------------
    # Run Pipeline
    # -------------------------------------------------

    result = run_pipeline(
        config=config,
        tracker=tracker,
    )

    # -------------------------------------------------
    # 1. Experiment must be initialized
    # -------------------------------------------------

    assert (
        FakeExperimentAdapter.initialized
        is True
    )

    # -------------------------------------------------
    # 2. Verify adapter call order
    # -------------------------------------------------

    assert [
        call[0]
        for call in calls
    ] == [
        "create_data_loaders",
        "create_model",
        "train_model",
    ]

    # -------------------------------------------------
    # 3. Tracker.save() must be called
    # -------------------------------------------------

    assert (
        tracker.save_called
        is True
    )

    # -------------------------------------------------
    # 4. Verify final pipeline result
    # -------------------------------------------------

    assert result == {
        "experiment_id": (
            "exp_pipeline_test"
        ),
        "best_epoch": 2,
        "best_metric": {
            "name": "val_accuracy",
            "value": 0.85,
        },
        "best_checkpoint": {
            "path": (
                "artifacts/checkpoints/"
                "exp_pipeline_test_best.pth"
            ),
            "epoch": 2,
            "metric_name": "val_accuracy",
            "metric_value": 0.85,
        },
        "test_metrics": {
            "loss": 0.30,
            "accuracy": 0.90,
        },
        "experiment_path": str(
            Path(
                "artifacts/experiments/"
                "exp_pipeline_test.json"
            )
        ),
    }
