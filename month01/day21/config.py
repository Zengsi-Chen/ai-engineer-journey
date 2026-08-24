from dataclasses import dataclass, field


@dataclass
class GeneralConfig:

    seed: int = 42

    device: str = "auto"


@dataclass
class DataConfig:

    source: str = "synthetic"

    batch_size: int = 32

    num_workers: int = 0

    train_ratio: float = 0.8

    val_ratio: float = 0.1

    test_ratio: float = 0.1


@dataclass
class ModelConfig:

    name: str = "mlp"

    params: dict = field(
        default_factory=dict
    )


@dataclass
class SchedulerConfig:

    enabled: bool = True

    type: str = "plateau"

    plateau_params: dict = field(
        default_factory=lambda: {
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6
        }
    )

    step_params: dict = field(
        default_factory=lambda: {
            "step_size": 100,
            "gamma": 0.1
        }
    )

    cosine_params: dict = field(
        default_factory=lambda: {
            "T_max": 500,
            "eta_min": 1e-6
        }
    )


@dataclass
class TrainingConfig:

    epochs: int = 20

    learning_rate: float = 0.001

    optimizer: str = "adam"

    optimizer_params: dict = field(
        default_factory=dict
    )

    resume: bool = False

    resume_experiment_id: str | None = None

    scheduler: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )


@dataclass
class CheckpointConfig:

    enabled: bool = True

    path: str = "artifacts/checkpoints"

    monitor: str = "val_loss"

    mode: str = "min"


@dataclass
class EarlyStoppingConfig:

    enabled: bool = True

    patience: int = 20

    min_delta: float = 0.001

    mode: str = "min"


@dataclass
class TaskConfig:

    task_type: str = "classification"

    num_classes: int = 2

    threshold: float = 0.5


@dataclass
class ArtifactConfig:

    root: str = "artifacts"


@dataclass
class Config:

    general: GeneralConfig = field(
        default_factory=GeneralConfig
    )

    data: DataConfig = field(
        default_factory=DataConfig
    )

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )

    checkpoint: CheckpointConfig = field(
        default_factory=CheckpointConfig
    )

    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )

    task: TaskConfig = field(
        default_factory=TaskConfig
    )

    artifacts: ArtifactConfig = field(
        default_factory=ArtifactConfig
    )