import torch.nn as nn
import pytest

from configs.pipeline_config import PipelineConfig
from src.model_adapter import create_model


def test_create_model_rejects_unknown_model():
    config = PipelineConfig()

    config.model.model_name = "unknown_model"

    with pytest.raises(
        ValueError,
        match="Only resnet18 is currently supported.",
    ):
        create_model(config)


def test_create_model_rejects_pretrained_model():
    config = PipelineConfig()

    config.model.pretrained = True

    with pytest.raises(
        ValueError,
        match="does not support pretrained weights",
    ):
        create_model(config)


def test_create_model_uses_num_classes(
    monkeypatch,
):
    config = PipelineConfig()

    config.model.num_classes = 5

    class FakeResNet18(nn.Module):

        def __init__(
            self,
            num_classes,
        ):
            super().__init__()

            self.num_classes = num_classes

    class FakeModelModule:
        ResNet18 = FakeResNet18

    monkeypatch.setattr(
        "src.model_adapter._load_day25_model_module",
        lambda: FakeModelModule,
    )

    model = create_model(config)

    assert isinstance(
        model,
        nn.Module,
    )

    assert model.num_classes == 5