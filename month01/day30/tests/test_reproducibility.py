from pathlib import Path

from configs.pipeline_config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
    CheckpointConfig,
)

from src.data_adapter import create_data_loaders
from src.model_adapter import create_model
from src.experiment_factory import (
    create_experiment_tracker,
)
from src.experiment_adapter import (
    ExperimentAdapter,
)
from src.training_adapter import train_model


def _create_config():
    return PipelineConfig(
        data=DataConfig(
            dataset_name="CIFAR10",
            batch_size=32,
            num_workers=0,
            max_train_samples=100,
            max_val_samples=50,
        ),
        model=ModelConfig(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
        ),
        training=TrainingConfig(
            epochs=1,
            learning_rate=0.001,
            device="cpu",
        ),
        experiment=ExperimentConfig(
            seed=42,
            experiment_name="day30_reproducibility_test",
        ),
        checkpoint=CheckpointConfig(
            output_dir="artifacts/checkpoints",
        ),
    )


def _run_training():
    config = _create_config()

    tracker = create_experiment_tracker()

    experiment_adapter = ExperimentAdapter(
        config=config,
        tracker=tracker,
    )

    experiment_adapter.initialize()

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_data_loaders(config)

    model = create_model(config)

    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
    )

    return {
        "training_history": (
            tracker.get_record()[
                "training_history"
            ]
        ),
        "best_epoch": result["best_epoch"],
        "best_metric": result["best_metric"],
        "test_metrics": result["test_metrics"],
    }


def test_same_seed_produces_same_results():
    first_result = _run_training()

    second_result = _run_training()

    assert (
        first_result["training_history"]
        == second_result["training_history"]
    )

    assert (
        first_result["best_epoch"]
        == second_result["best_epoch"]
    )

    assert (
        first_result["best_metric"]
        == second_result["best_metric"]
    )
