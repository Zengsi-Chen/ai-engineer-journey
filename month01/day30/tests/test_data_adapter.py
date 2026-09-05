from unittest.mock import Mock

import pytest

from configs.pipeline_config import PipelineConfig
from src.data_adapter import create_data_loaders


def test_create_data_loaders_rejects_unknown_dataset():
    config = PipelineConfig()

    config.data.dataset_name = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="Only CIFAR10 is currently supported",
    ):
        create_data_loaders(config)


def test_create_data_loaders_uses_config_values(
    monkeypatch,
):
    config = PipelineConfig()

    config.data.batch_size = 32
    config.data.max_train_samples = 100
    config.data.max_val_samples = 20
    config.data.num_workers = 0
    config.experiment.seed = 123

    fake_train_loader = Mock()
    fake_validation_loader = Mock()
    fake_test_loader = Mock()

    fake_create_loaders = Mock(
        return_value=(
            fake_train_loader,
            fake_validation_loader,
            fake_test_loader,
        )
    )

    fake_module = Mock()

    fake_module.create_cifar10_loaders = (
        fake_create_loaders
    )

    monkeypatch.setattr(
        "src.data_adapter._load_day25_data_module",
        lambda: fake_module,
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_data_loaders(config)

    assert train_loader is fake_train_loader

    assert (
        validation_loader
        is fake_validation_loader
    )

    assert test_loader is fake_test_loader

    fake_create_loaders.assert_called_once_with(
        batch_size=32,
        train_samples=100,
        validation_samples=20,
        seed=123,
        num_workers=0,
    )

