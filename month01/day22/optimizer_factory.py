import torch.optim as optim


def create_optimizer(
    model,
    config
):

    if config.optimizer == "adam":

        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            **config.optimizer_params
        )

    if config.optimizer == "sgd":

        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            **config.optimizer_params
        )

    raise ValueError(
        f"Unknown optimizer: {config.optimizer}"
    )