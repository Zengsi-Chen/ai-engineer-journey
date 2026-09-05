from configs.pipeline_config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)


def test_pipeline_config_defaults():
    config = PipelineConfig()

    assert isinstance(
        config.data,
        DataConfig,
    )

    assert isinstance(
        config.model,
        ModelConfig,
    )

    assert isinstance(
        config.training,
        TrainingConfig,
    )

    assert isinstance(
        config.experiment,
        ExperimentConfig,
    )


def test_pipeline_config_uses_default_values():
    config = PipelineConfig()

    assert config.data.dataset_name == "CIFAR10"
    assert config.data.batch_size == 64
    assert config.data.max_train_samples == 1000
    assert config.data.max_val_samples == 200

    assert config.model.model_name == "resnet18"
    assert config.model.num_classes == 10

    assert config.training.epochs == 2
    assert config.training.learning_rate == 0.001
    assert config.training.device == "cpu"

    assert config.experiment.seed == 42
    assert (
        config.experiment.experiment_name
        == "day30_capstone"
    )


def test_pipeline_configs_do_not_share_nested_objects():
    config_a = PipelineConfig()
    config_b = PipelineConfig()

    config_a.data.batch_size = 128

    assert config_b.data.batch_size == 64


def test_checkpoint_config_has_default_output_dir():
    config = PipelineConfig()

    assert config.checkpoint.output_dir == "artifacts/checkpoints"


def test_checkpoint_output_dir_can_be_customized():
    config = PipelineConfig()
    config.checkpoint.output_dir = "custom/checkpoints"

    assert config.checkpoint.output_dir == "custom/checkpoints"