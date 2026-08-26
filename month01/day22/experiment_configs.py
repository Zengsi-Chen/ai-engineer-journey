from config import (
    Config,
    DataConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
)


def create_binary_mlp_config():

    return Config(

        data=DataConfig(
            source="binary",
            batch_size=16
        ),

        model=ModelConfig(
            name="mlp",
            input_dim=1,
            hidden_dim=16,
            output_dim=1
        ),

        task=TaskConfig(
            task_type="binary_classification",
            threshold=0.5
        ),

        training=TrainingConfig(
            epochs=20,
            learning_rate=0.001
        )
    )


def create_mnist_cnn_config():

    return Config(

        data=DataConfig(
            source="mnist",
            batch_size=64
        ),

        model=ModelConfig(
            name="cnn",
            input_channels=1,
            num_classes=10
        ),

        task=TaskConfig(
            task_type="multiclass_classification"
        ),

        training=TrainingConfig(
            epochs=5,
            learning_rate=0.001
        )
    )


def create_invalid_mnist_binary_config():

    return Config(

        data=DataConfig(
            source="mnist",
            batch_size=64
        ),

        model=ModelConfig(
            name="mlp",
            input_dim=1,
            hidden_dim=16,
            output_dim=1
        ),

        task=TaskConfig(
            task_type="binary_classification",
            threshold=0.5
        ),

        training=TrainingConfig(
            epochs=20,
            learning_rate=0.001
        )
    )


def create_invalid_binary_cnn_config():

    return Config(

        data=DataConfig(
            source="binary",
            batch_size=16
        ),

        model=ModelConfig(
            name="cnn",
            input_channels=1,
            num_classes=10
        ),

        task=TaskConfig(
            task_type="multiclass_classification"
        ),

        training=TrainingConfig(
            epochs=20,
            learning_rate=0.001
        )
    )