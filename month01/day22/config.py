from dataclasses import dataclass, field


@dataclass
class GeneralConfig:

    seed: int = 42

    device: str = "auto"


@dataclass
class DataConfig:

    source: str = "binary"

    batch_size: int = 32

    num_workers: int = 0

    train_ratio: float = 0.8

    val_ratio: float = 0.1

    test_ratio: float = 0.1


@dataclass
class ModelConfig:

    name: str = "cnn"

    input_dim: int | None = None

    hidden_dim: int | None = None

    output_dim: int | None = None

    input_channels: int = 1

    num_classes: int = 10


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

    epochs: int = 5

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

    task_type: str = "multiclass_classification"

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


def validate_config(config):

    task_type = config.task.task_type
    model_name = config.model.name.lower()

    data_source = config.data.source

    # =========================
    # Data / Model compatibility
    # =========================

    if data_source == "mnist":

        if model_name != "cnn":

            raise ValueError(
                "MNIST dataset requires "
                "model='cnn'"
            )

        if config.model.input_channels != 1:

            raise ValueError(
                "MNIST requires "
                "input_channels=1"
            )

        if config.model.num_classes != 10:

            raise ValueError(
                "MNIST requires "
                "num_classes=10"
            )

    elif data_source == "binary":

        if model_name != "mlp":

            raise ValueError(
                "Binary dataset requires "
                "model='mlp'"
            )

        if config.model.input_dim != 1:

            raise ValueError(
                "Binary dataset requires "
                "input_dim=1"
            )

        if config.model.output_dim != 1:

            raise ValueError(
                "Binary classification requires "
                "output_dim=1"
            )

    else:

        raise ValueError(
            f"Unknown data source: {data_source}"
        )

    # =========================
    # Data / Task compatibility
    # =========================

    if data_source == "mnist":

        if task_type != "multiclass_classification":

            raise ValueError(
                "MNIST requires "
                "multiclass_classification"
            )

    elif data_source == "binary":

        if task_type != "binary_classification":

            raise ValueError(
                "Binary dataset requires "
                "binary_classification"
            )

    # =========================
    # Binary Classification
    # =========================

    if task_type == "binary_classification":

        if model_name == "mlp":

            if config.model.output_dim != 1:

                raise ValueError(
                    "Binary classification with MLP "
                    "requires output_dim=1"
                )

        elif model_name == "cnn":

            if config.model.num_classes != 1:

                raise ValueError(
                    "Binary classification with CNN "
                    "requires num_classes=1"
                )

        else:

            raise ValueError(
                f"Unsupported model for binary "
                f"classification: {model_name}"
            )

    # =========================
    # Multiclass Classification
    # =========================

    elif task_type == "multiclass_classification":

        if model_name == "cnn":

            if config.model.num_classes < 2:

                raise ValueError(
                    "Multiclass classification with CNN "
                    "requires num_classes >= 2"
                )

        elif model_name == "mlp":

            if config.model.output_dim < 2:

                raise ValueError(
                    "Multiclass classification with MLP "
                    "requires output_dim >= 2"
                )

        else:

            raise ValueError(
                f"Unsupported model for multiclass "
                f"classification: {model_name}"
            )

    # =========================
    # Regression
    # =========================

    elif task_type == "regression":

        if model_name == "mlp":

            if config.model.output_dim != 1:

                raise ValueError(
                    "Regression with MLP "
                    "requires output_dim=1"
                )

        else:

            raise ValueError(
                f"Unsupported model for regression: "
                f"{model_name}"
            )

    # =========================
    # Unknown Task
    # =========================

    else:

        raise ValueError(
            f"Unknown task type: {task_type}"
        )