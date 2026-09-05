import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _load_day25_training_module():
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
        "training_utils"
    )


def evaluate_model(
    model,
    dataloader,
    config,
):
    """
    Evaluate a model on a dataset.

    This function reuses the Day 25 evaluation
    implementation instead of duplicating it.
    """

    device = torch.device(
        config.training.device
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    training_module = (
        _load_day25_training_module()
    )

    loss, accuracy = (
        training_module.evaluate(
            model,
            dataloader,
            criterion,
            device,
        )
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
    }