import json
from pathlib import Path

import torch

from configs.pipeline_config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
    CheckpointConfig,
)

from src.checkpoint_adapter import (
    load_checkpoint,
)

from src.model_adapter import (
    create_model,
)


def _load_latest_experiment():
    experiment_dir = Path(
        "artifacts/experiments"
    )

    experiment_files = list(
        experiment_dir.glob("*.json")
    )

    assert experiment_files, (
        "No experiment JSON file was found."
    )

    experiment_path = max(
        experiment_files,
        key=lambda path: path.stat().st_mtime,
    )

    with open(
        experiment_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_best_checkpoint_can_be_reloaded():
    record = _load_latest_experiment()

    checkpoint_info = record[
        "best_checkpoint"
    ]

    checkpoint_path = Path(
        checkpoint_info["path"]
    )

    assert checkpoint_path.exists()

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
            experiment_name="checkpoint_recovery_test",
        ),
        checkpoint=CheckpointConfig(
            output_dir="artifacts/checkpoints",
        ),
    )

    model = create_model(config)

    model = load_checkpoint(
        model=model,
        path=checkpoint_path,
        device=torch.device("cpu"),
    )

    model.eval()

    inputs = torch.randn(
        2,
        3,
        32,
        32,
    )

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (
        2,
        config.model.num_classes,
    )

    assert torch.isfinite(outputs).all()
