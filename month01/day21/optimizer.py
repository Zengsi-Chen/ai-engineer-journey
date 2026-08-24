import torch


def create_optimizer(
    model,
    config
):

    optimizer_name = (
        config.training.optimizer.lower()
    )

    learning_rate = (
        config.training.learning_rate
    )

    optimizer_params = (
        config.training.optimizer_params
    )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            **optimizer_params
        )

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            **optimizer_params
        )

    if optimizer_name == "sgd":

        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            **optimizer_params
        )

    raise ValueError(
        f"Unknown optimizer: "
        f"{optimizer_name}"
    )


def create_scheduler(
    optimizer,
    config
):

    scheduler_config = (
        config.training.scheduler
    )

    if not scheduler_config.enabled:

        return None

    scheduler_name = (
        scheduler_config.type.lower()
    )


    if scheduler_name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            **scheduler_config.step_params
        )


    if scheduler_name == "cosine":

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **scheduler_config.cosine_params
        )


    if scheduler_name == "plateau":

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_config.plateau_params
        )


    raise ValueError(
        f"Unknown scheduler: "
        f"{scheduler_name}"
    )