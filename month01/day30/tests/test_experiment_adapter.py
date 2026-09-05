from configs.pipeline_config import PipelineConfig
from src.experiment_adapter import ExperimentAdapter


class FakeTracker:

    def __init__(self):
        self.seed = None
        self.metadata = {}
        self.hyperparameters = {}

    def set_reproducibility(self, seed):
        self.seed = seed

    def log_metadata(self, metadata):
        self.metadata.update(metadata)

    def log_hyperparameters(self, params):
        self.hyperparameters.update(params)


def test_initialize_sets_seed():
    config = PipelineConfig()
    tracker = FakeTracker()

    adapter = ExperimentAdapter(
        config,
        tracker,
    )

    adapter.initialize()

    assert (
        tracker.seed
        == config.experiment.seed
    )


def test_initialize_logs_experiment_name():
    config = PipelineConfig()
    tracker = FakeTracker()

    adapter = ExperimentAdapter(
        config,
        tracker,
    )

    adapter.initialize()

    assert (
        tracker.metadata["experiment_name"]
        == config.experiment.experiment_name
    )


def test_initialize_logs_hyperparameters():
    config = PipelineConfig()
    tracker = FakeTracker()

    adapter = ExperimentAdapter(
        config,
        tracker,
    )

    adapter.initialize()

    assert (
        tracker.hyperparameters["dataset_name"]
        == "CIFAR10"
    )

    assert (
        tracker.hyperparameters["model_name"]
        == "resnet18"
    )

    assert (
        tracker.hyperparameters["learning_rate"]
        == 0.001
    )


def test_initialize_returns_tracker():
    config = PipelineConfig()
    tracker = FakeTracker()

    adapter = ExperimentAdapter(
        config,
        tracker,
    )

    result = adapter.initialize()

    assert result is tracker