import torch

from dataset import create_data_pipeline
from model import create_model
from loss import create_loss

from optimizer import (
    create_optimizer,
    create_scheduler
)

from checkpoint import CheckpointManager
from early_stopping import EarlyStopping
from trainer import Trainer


class Pipeline:

    def __init__(
        self,
        config
    ):

        self.config = config

        self.device = (
            self._resolve_device()
        )

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.model = None
        self.loss_fn = None

        self.optimizer = None
        self.scheduler = None

        self.checkpoint = None
        self.early_stopping = None

        self.trainer = None

        self.start_epoch = 0


    # =========================
    # Device
    # =========================

    def _resolve_device(self):

        if self.config.general.device == "auto":

            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return self.config.general.device


    # =========================
    # Data
    # =========================

    def _create_data(self):

        (
            self.train_loader,
            self.val_loader,
            self.test_loader
        ) = create_data_pipeline(
            config=self.config
        )


    # =========================
    # Model
    # =========================

    def _create_model(self):

        self.model = create_model(
            self.config.model
        )

        self.model.to(
            self.device
        )


    # =========================
    # Loss
    # =========================

    def _create_loss(self):

        self.loss_fn = create_loss(
            self.config
        )


    # =========================
    # Optimizer
    # =========================

    def _create_optimizer(self):

        self.optimizer = create_optimizer(
            self.model,
            self.config
        )


    # =========================
    # Scheduler
    # =========================

    def _create_scheduler(self):

        self.scheduler = create_scheduler(
            self.optimizer,
            self.config
        )


    # =========================
    # Checkpoint
    # =========================

    def _create_checkpoint(self):

        if not self.config.checkpoint.enabled:

            self.checkpoint = None

            return

        self.checkpoint = CheckpointManager(
            path=self.config.checkpoint.path,
            monitor=self.config.checkpoint.monitor,
            mode=self.config.checkpoint.mode
        )


    # =========================
    # Early Stopping
    # =========================

    def _create_early_stopping(self):

        if not self.config.early_stopping.enabled:

            self.early_stopping = None

            return

        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
            mode=self.config.early_stopping.mode
        )


    # =========================
    # Trainer
    # =========================

    def _create_trainer(self):

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            scheduler=self.scheduler,
            checkpoint=self.checkpoint,
            early_stopping=self.early_stopping,
            device=self.device
        )


    # =========================
    # Setup
    # =========================

    def setup(self):

        self._create_data()

        self._create_model()

        self._create_loss()

        self._create_optimizer()

        self._create_scheduler()

        self._create_checkpoint()

        self._create_early_stopping()

        self._create_trainer()


    # =========================
    # Load Checkpoint
    # =========================

    def _load_checkpoint(self):

        if self.checkpoint is None:

            self.start_epoch = 0

            return

        self.start_epoch = (
            self.checkpoint.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping
            )
        )


    # =========================
    # Train
    # =========================

    def train(self):

        self._load_checkpoint()

        print(
            f"Resume from epoch: "
            f"{self.start_epoch}"
        )

        history = self.trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.config.training.epochs,
            start_epoch=self.start_epoch
        )

        return history



if __name__ == "__main__":

    from config import Config

    config = Config()

    config.model.name = "mlp"

    config.training.scheduler.type = "cosine"

    config.model.params = {
        "input_dim": 1,
        "hidden_dim": 16,
        "output_dim": 1
    }

    pipeline = Pipeline(
        config
    )

    pipeline.setup()

    print(
        "Device:",
        pipeline.device
    )

    print(
        "Optimizer:",
        pipeline.optimizer
    )

    print(
        "Scheduler:",
        pipeline.scheduler
    )

    print(
        "Checkpoint:",
        pipeline.checkpoint
    )

    print(
        "EarlyStopping:",
        pipeline.early_stopping
    )

    print(
        "Trainer:",
        pipeline.trainer
    )