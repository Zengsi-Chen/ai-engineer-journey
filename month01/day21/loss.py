import torch.nn as nn


def create_loss(config):

    task_type = (
        config.task.task_type.lower()
    )


    if task_type == "classification":

        return nn.BCEWithLogitsLoss()


    if task_type == "regression":

        return nn.MSELoss()


    raise ValueError(
        f"Unknown task type: {task_type}"
    )