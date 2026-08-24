import torch

from training_state import EpochResult
from training_state import (
    EpochResult,
    TrainingHistory
)


def move_to_device(
    data,
    device
):

    if isinstance(
        data,
        torch.Tensor
    ):

        return data.to(device)


    if isinstance(
        data,
        dict
    ):

        return {
            key: move_to_device(
                value,
                device
            )
            for key, value in data.items()
        }


    if isinstance(
        data,
        list
    ):

        return [
            move_to_device(
                item,
                device
            )
            for item in data
        ]


    if isinstance(
        data,
        tuple
    ):

        return tuple(
            move_to_device(
                item,
                device
            )
            for item in data
        )


    return data


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        checkpoint=None,
        early_stopping=None,
        device="cpu"
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.device = device

        self.model.to(self.device)


    def train_epoch(
        self,
        train_loader
    ):

        self.model.train()

        total_loss = 0.0

        for inputs, targets in train_loader:

            inputs = move_to_device(
                inputs,
                self.device
            )

            targets = move_to_device(
                targets,
                self.device
            )

            self.optimizer.zero_grad()

            output = self.model(
                inputs
            )

            loss = self.loss_fn(
                output,
                targets
            )

            loss.backward()

            self.optimizer.step()

            batch_size = (
                targets.size(0)
                if isinstance(
                    targets,
                    torch.Tensor
                )
                else inputs.size(0)
            )


            total_loss += (
                loss.item()
                * batch_size
            )


        epoch_loss = (
            total_loss
            / len(train_loader.dataset)
        )

        return epoch_loss

    def validate(
        self,
        val_loader
    ):

        self.model.eval()

        total_loss = 0.0


        with torch.no_grad():

            for inputs, targets in val_loader:

                inputs = move_to_device(
                    inputs,
                    self.device
                )

                targets = move_to_device(
                    targets,
                    self.device
                )


                output = self.model(
                    inputs
                )


                loss = self.loss_fn(
                    output,
                    targets
                )


                batch_size = (
                    targets.size(0)
                    if isinstance(
                        targets,
                        torch.Tensor
                    )
                    else inputs.size(0)
                )


                total_loss += (
                    loss.item()
                    * batch_size
                )


        val_loss = (
            total_loss
            / len(val_loader.dataset)
        )


        return val_loss
        

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        start_epoch=0
    ):

        history = TrainingHistory(
            train_loss=[],
            val_loss=[],
            learning_rate=[]
        )


        for epoch in range(
            start_epoch,
            epochs
        ):

            # =========================
            # 1. Training
            # =========================

            train_loss = self.train_epoch(
                train_loader
            )


            # =========================
            # 2. Validation
            # =========================

            val_loss = self.validate(
                val_loader
            )


            # =========================
            # 3. Scheduler
            # =========================

            if self.scheduler is not None:

                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):

                    self.scheduler.step(
                        val_loss
                    )

                else:

                    self.scheduler.step()


            # =========================
            # 4. Learning Rate
            # =========================

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )


            # =========================
            # 5. Early Stopping
            # =========================

            should_stop = False

            if self.early_stopping is not None:

                should_stop = (
                    self.early_stopping.step(
                        val_loss
                    )
                )


            # =========================
            # 6. Checkpoint
            # =========================

            is_best = False

            if self.checkpoint is not None:

                is_best = (
                    self.checkpoint.save_best(
                        model=self.model,
                        current_value=val_loss,
                        epoch=epoch + 1
                    )
                )

                self.checkpoint.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    current_value=val_loss,
                    early_stopping=self.early_stopping
                )


            # =========================
            # 7. Epoch Result
            # =========================

            epoch_result = EpochResult(

                epoch=epoch + 1,

                train_loss=train_loss,

                val_loss=val_loss,

                learning_rate=current_lr,

                is_best=is_best,

                should_stop=should_stop
            )


            # =========================
            # 8. History
            # =========================

            history.add(
                epoch_result
            )


            # =========================
            # 9. Logging
            # =========================
            if (epoch + 1) % 10 == 0:

                print(
                    f"Epoch [{epoch_result.epoch}/{epochs}] "
                    f"Train Loss: "
                    f"{epoch_result.train_loss:.4f} "
                    f"Val Loss: "
                    f"{epoch_result.val_loss:.4f} "
                    f"LR: "
                    f"{epoch_result.learning_rate:.6f}"
                )


            if epoch_result.is_best:

                print(
                    "New best model saved."
                )


            # =========================
            # 10. Early Stop
            # =========================

            if epoch_result.should_stop:

                print(
                    "Early stopping triggered."
                )

                break


        return history