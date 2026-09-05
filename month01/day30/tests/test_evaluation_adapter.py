from unittest.mock import Mock

import torch

from configs.pipeline_config import PipelineConfig
from src.evaluation_adapter import evaluate_model


def test_evaluate_model_returns_metrics(
    monkeypatch,
):
    config = PipelineConfig()

    fake_model = Mock()

    fake_model.to.return_value = fake_model

    fake_evaluate = Mock(
        return_value=(0.35, 0.82)
    )

    fake_training_module = Mock()

    fake_training_module.evaluate = (
        fake_evaluate
    )

    monkeypatch.setattr(
        "src.evaluation_adapter._load_day25_training_module",
        lambda: fake_training_module,
    )

    result = evaluate_model(
        model=fake_model,
        dataloader=Mock(),
        config=config,
    )

    assert result == {
        "loss": 0.35,
        "accuracy": 0.82,
    }

    fake_evaluate.assert_called_once()


def test_evaluate_model_uses_config_device(
    monkeypatch,
):
    config = PipelineConfig()

    config.training.device = "cpu"

    fake_model = Mock()

    fake_model.to.return_value = fake_model

    fake_evaluate = Mock(
        return_value=(0.5, 0.75)
    )

    fake_training_module = Mock()

    fake_training_module.evaluate = (
        fake_evaluate
    )

    monkeypatch.setattr(
        "src.evaluation_adapter._load_day25_training_module",
        lambda: fake_training_module,
    )

    evaluate_model(
        model=fake_model,
        dataloader=Mock(),
        config=config,
    )

    args = fake_evaluate.call_args.args

    assert args[3] == torch.device("cpu")


