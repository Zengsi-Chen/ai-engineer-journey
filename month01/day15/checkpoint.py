import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    early_stop_counter
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "early_stop_counter": early_stop_counter,
    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler
):

    checkpoint = torch.load(
        path,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] +1

    best_val_loss = checkpoint["best_val_loss"]

    early_stop_counter = checkpoint["early_stop_counter"]

    return (
        start_epoch,
        best_val_loss,
        early_stop_counter
    )