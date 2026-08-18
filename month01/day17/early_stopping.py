import torch


class EarlyStopping:

    def __init__(
        self,
        patience=10,
        min_delta=0.001,
        checkpoint_path="checkpoint.pth",
        best_model_path="best_model.pth"
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path

        self.best_val_loss = float("inf")
        self.counter = 0

    def __call__(
        self,
        val_loss,
        model,
        optimizer,
        epoch
    ):

        # =========================
        # 1. Check whether this is
        #    the best model
        # =========================

        if val_loss < self.best_val_loss - self.min_delta:

            self.best_val_loss = val_loss
            self.counter = 0

            # -------------------------
            # Save Best Model
            # -------------------------

            torch.save(
                model.state_dict(),
                self.best_model_path
            )

            print(
                f"Best model saved: "
                f"Val Loss = {val_loss:.4f}"
            )

        else:

            self.counter += 1

        # =========================
        # 2. Always save checkpoint
        # =========================

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "early_stopping_counter": self.counter
            },
            self.checkpoint_path
        )

        # =========================
        # 3. Early Stopping
        # =========================

        return self.counter >= self.patience