import torch
import torch.nn as nn
import pytest

from src.checkpoint_adapter import (
    save_checkpoint,
    load_checkpoint,
)


def test_save_checkpoint_creates_file(tmp_path):
    model = nn.Linear(4, 2)

    checkpoint_path = tmp_path / "checkpoints" / "best.pth"

    result = save_checkpoint(
        model,
        checkpoint_path,
    )

    assert result == checkpoint_path
    assert checkpoint_path.exists()


def test_load_checkpoint_restores_model(tmp_path):
    model = nn.Linear(4, 2)

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(2.0)

    checkpoint_path = tmp_path / "best.pth"

    save_checkpoint(
        model,
        checkpoint_path,
    )

    restored_model = nn.Linear(4, 2)

    load_checkpoint(
        restored_model,
        checkpoint_path,
        torch.device("cpu"),
    )

    assert torch.equal(
        model.weight,
        restored_model.weight,
    )

    assert torch.equal(
        model.bias,
        restored_model.bias,
    )


def test_load_checkpoint_raises_for_missing_file(tmp_path):
    model = nn.Linear(4, 2)

    checkpoint_path = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            model,
            checkpoint_path,
            torch.device("cpu"),
        )