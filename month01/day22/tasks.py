from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Task(ABC):

    @abstractmethod
    def create_loss(self):
        pass

    @abstractmethod
    def predict(self, outputs):
        pass


class BinaryClassificationTask(Task):

    def __init__(self, threshold=0.5):

        self.threshold = threshold

    def create_loss(self):

        return nn.BCEWithLogitsLoss()

    def predict(self, outputs):

        probability = torch.sigmoid(outputs)

        prediction = (
            probability > self.threshold
        ).long()

        return prediction


class MulticlassClassificationTask(Task):

    def create_loss(self):

        return nn.CrossEntropyLoss()

    def predict(self, outputs):

        return outputs.argmax(dim=1)


class RegressionTask(Task):

    def create_loss(self):

        return nn.MSELoss()

    def predict(self, outputs):

        return outputs