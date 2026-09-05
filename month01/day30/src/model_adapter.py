import importlib
import sys
from pathlib import Path

import torch.nn as nn


def _load_day25_model_module():
    project_root = (
        Path(__file__).resolve().parents[3]
    )

    day25_path = (
        project_root
        / "month01"
        / "day25"
    )

    day25_path_str = str(day25_path)

    if day25_path_str not in sys.path:
        sys.path.insert(
            0,
            day25_path_str,
        )

    return importlib.import_module(
        "resnet18"
    )


def create_model(config) -> nn.Module:
    """
    Create a model using the existing
    Day 25 model implementation.
    """

    model_name = (
        config.model.model_name.lower()
    )

    if model_name != "resnet18":
        raise ValueError(
            "Only resnet18 is currently supported."
        )

    if config.model.pretrained:
        raise ValueError(
            "Day 25 ResNet18 does not support "
            "pretrained weights."
        )

    model_module = (
        _load_day25_model_module()
    )

    model = model_module.ResNet18(
        num_classes=config.model.num_classes
    )

    return model