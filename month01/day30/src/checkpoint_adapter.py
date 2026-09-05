from pathlib import Path

import torch


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)

    return path


def load_checkpoint(model, path, device):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    state_dict = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    return model