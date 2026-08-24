import os
import torch


class CheckpointManager:

    def __init__(
        self,
        path,
        monitor="val_loss",
        mode="min"
    ):

        self.path = path
        self.monitor = monitor
        self.mode = mode

        self.best_value = None
        self.best_epoch = None

        os.makedirs(
            self.path,
            exist_ok=True
        )

        self.best_model_path = os.path.join(
            self.path,
            "best_model.pth"
        )


    def is_better(
        self,
        current_value
    ):

        if self.best_value is None:
            return True

        if self.mode == "min":
            return current_value < self.best_value

        if self.mode == "max":
            return current_value > self.best_value

        raise ValueError(
            f"Unknown checkpoint mode: {self.mode}"
        )


    def save_best(
        self,
        model,
        current_value,
        epoch
    ):

        if not self.is_better(
            current_value
        ):
            return False

        self.best_value = current_value
        self.best_epoch = epoch

        torch.save(
            model.state_dict(),
            self.best_model_path
        )

        return True

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        current_value,
        early_stopping=None
    ):

        checkpoint = {

            "epoch": epoch,

            "best_epoch":
                self.best_epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "best_value":
                self.best_value,

            "current_value":
                current_value,

            "early_stopping_state": (
                {
                    "best_value":
                        early_stopping.best_value,

                    "counter":
                        early_stopping.counter
                }
                if early_stopping is not None
                else None
            )
        }

        checkpoint_path = os.path.join(
            self.path,
            "checkpoint.pth"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        


    def load_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        early_stopping=None
    ):

        checkpoint_path = os.path.join(
            self.path,
            "checkpoint.pth"
        )

        

        if not os.path.exists(checkpoint_path):

            return 0

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "MODEL LOADED SUCCESSFULLY"
        )

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.best_value = checkpoint.get(
            "best_value"
        )

        self.best_epoch = checkpoint.get(
            "best_epoch"
        )

        early_stopping_state = checkpoint.get(
            "early_stopping_state"
        )

        if (
            early_stopping is not None
            and early_stopping_state is not None
        ):

            early_stopping.best_value = (
                early_stopping_state["best_value"]
            )

            early_stopping.counter = (
                early_stopping_state["counter"]
            )

        return checkpoint["epoch"]

    def load_best_model(
        self,
        model,
        device="cpu"
    ):

        best_model_path = os.path.join(
            self.path,
            "best_model.pth"
        )

        if not os.path.exists(
            best_model_path
        ):

            raise FileNotFoundError(
                f"Best model not found: "
                f"{best_model_path}"
            )

        state_dict = torch.load(
            best_model_path,
            map_location=device
        )

        model.load_state_dict(
            state_dict
        )

        model.to(device)

        return model