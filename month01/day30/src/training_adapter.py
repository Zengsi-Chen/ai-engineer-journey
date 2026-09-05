import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.checkpoint_adapter import (
    load_checkpoint,
    save_checkpoint,
)

from src.evaluation_adapter import (
    evaluate_model,
)


def _load_day25_training_module():
    project_root = (
        Path(__file__).resolve().parents[3]
    )

    day25_path = (
        project_root
        / "month01"
        / "day25"
    )

    day25_path_str = str(day25_path)

    if day25_path_str not in sys.path:
        sys.path.insert(
            0,
            day25_path_str,
        )

    return importlib.import_module(
        "training_utils"
    )


def train_model(
    model,
    train_loader,
    validation_loader,
    test_loader,
    config,
    tracker,
):
    device = torch.device(
        config.training.device
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
    )

    training_module = (
        _load_day25_training_module()
    )

    train_one_epoch = (
        training_module.train_one_epoch
    )

    checkpoint_dir = Path(
        config.checkpoint.output_dir
    )

    experiment_id = (
        tracker.get_record()["experiment_id"]
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{experiment_id}_best.pth"
    )

    for epoch in range(
        1,
        config.training.epochs + 1,
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_metrics = evaluate_model(
            model=model,
            dataloader=validation_loader,
            config=config,
        )

        val_loss = validation_metrics["loss"]

        val_accuracy = (
            validation_metrics["accuracy"]
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }

        tracker.log_epoch(metrics)

        record = tracker.get_record()

        if record["best_epoch"] == epoch:
            saved_path = save_checkpoint(
                model,
                checkpoint_path,
            )

            tracker.set_best_checkpoint(
                str(saved_path)
            )

    best_checkpoint = (
        tracker.get_record()["best_checkpoint"]
    )

    if best_checkpoint is None:
        raise RuntimeError(
            "No best checkpoint was created."
        )

    checkpoint_path = (
        best_checkpoint["path"]
    )

    model = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        config=config,
    )

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "best_epoch": (
            tracker.get_record()["best_epoch"]
        ),
        "best_metric": (
            tracker.get_record()["best_metric"]
        ),
        "best_checkpoint": best_checkpoint,
        "test_metrics": test_metrics,
    }