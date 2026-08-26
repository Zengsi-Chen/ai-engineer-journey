import torch

from abc import ABC, abstractmethod


class Metrics(ABC):

    @abstractmethod
    def compute(self, predictions, targets):
        pass


class ClassificationMetrics(Metrics):

    def compute(self, predictions, targets):

        correct = (
            predictions == targets
        ).sum().item()

        total = targets.numel()

        accuracy = correct / total

        return {
            "accuracy": accuracy
        }


class RegressionMetrics(Metrics):

    def compute(
        self,
        predictions,
        targets
    ):

        error = predictions - targets

        mae = error.abs().mean()

        mse = (error ** 2).mean()

        rmse = torch.sqrt(mse)

        return {
            "mae": mae.item(),
            "mse": mse.item(),
            "rmse": rmse.item()
        }
