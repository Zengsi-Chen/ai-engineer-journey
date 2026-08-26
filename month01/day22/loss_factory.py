import torch.nn as nn


def create_loss(task):

    if task == "binary_classification":

        return nn.BCEWithLogitsLoss()

    if task == "multiclass_classification":

        return nn.CrossEntropyLoss()

    if task == "regression":

        return nn.MSELoss()

    raise ValueError(
        f"Unknown task: {task}"
    )