from training import train_one_epoch
from evaluation import evaluate


class MLPipeline:

    def __init__(
        self,
        model,
        task,
        optimizer,
        train_loader,
        val_loader,
        metrics
    ):

        self.model = model
        self.task = task
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.metrics = metrics

    def train_epoch(self):

        return train_one_epoch(
            model=self.model,
            dataloader=self.train_loader,
            task=self.task,
            optimizer=self.optimizer
        )

    def evaluate(self):

        return evaluate(
            model=self.model,
            dataloader=self.val_loader,
            task=self.task,
            metrics=self.metrics
        )