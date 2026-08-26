from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, directory="artifacts"):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_model_path = (
            self.directory / "best_model.pth"
        )

    def save_best(
        self,
        model,
        optimizer,
        epoch,
        best_val_accuracy,
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
        }

        torch.save(
            checkpoint,
            self.best_model_path,
        )

        print(
            f"Best model saved: "
            f"epoch={epoch}, "
            f"val_accuracy={best_val_accuracy:.4f}"
        )

    def load_best(
        self,
        model,
        optimizer=None,
    ):
        checkpoint = torch.load(
            self.best_model_path,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return checkpoint