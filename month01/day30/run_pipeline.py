from configs.pipeline_config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
    CheckpointConfig,
)

from src.experiment_factory import (
    create_experiment_tracker,
)

from src.pipeline import run_pipeline


def main():
    config = PipelineConfig(
        data=DataConfig(
            dataset_name="CIFAR10",
            batch_size=64,
            num_workers=0,
            max_train_samples=1000,
            max_val_samples=200,
        ),
        model=ModelConfig(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
        ),
        training=TrainingConfig(
            epochs=2,
            learning_rate=0.001,
            device="cpu",
        ),
        experiment=ExperimentConfig(
            seed=42,
            experiment_name="day30_capstone",
        ),
        checkpoint=CheckpointConfig(
            output_dir="artifacts/checkpoints",
        ),
    )

    tracker = create_experiment_tracker()

    result = run_pipeline(
        config=config,
        tracker=tracker,
    )

    print()
    print("=" * 60)
    print("DAY 30 CAPSTONE RESULT")
    print("=" * 60)
    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Best Epoch: {result['best_epoch']}")
    print(f"Best Metric: {result['best_metric']}")
    print(f"Best Checkpoint: {result['best_checkpoint']}")
    print(f"Test Metrics: {result['test_metrics']}")
    print(f"Experiment JSON: {result['experiment_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

