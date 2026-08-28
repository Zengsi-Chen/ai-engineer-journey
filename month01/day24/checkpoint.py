from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_model_path = (
            self.checkpoint_dir / "best_model.pth"
        )

    def save_best_model(
        self,
        model,
        epoch,
        metric,
    ):
        checkpoint = {
            "epoch": epoch,
            "metric": metric,
            "model_state_dict": model.state_dict(),
        }

        torch.save(
            checkpoint,
            self.best_model_path,
        )

    def load_best_model(
        self,
        model,
        device,
    ):
        checkpoint = torch.load(
            self.best_model_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        return checkpoint