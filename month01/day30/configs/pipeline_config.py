from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset_name: str = "CIFAR10"
    batch_size: int = 64
    num_workers: int = 0
    max_train_samples: int | None = 1000
    max_val_samples: int | None = 200


@dataclass
class ModelConfig:
    model_name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 0.001
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    seed: int = 42
    experiment_name: str = "day30_capstone"


@dataclass
class CheckpointConfig:
    output_dir: str = "artifacts/checkpoints"


@dataclass
class PipelineConfig:
    data: DataConfig = field(
        default_factory=DataConfig
    )

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )

    experiment: ExperimentConfig = field(
        default_factory=ExperimentConfig
    )

    checkpoint: CheckpointConfig = field(
        default_factory=CheckpointConfig
    )