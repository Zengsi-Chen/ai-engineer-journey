import time
import torch

from dataclasses import dataclass
from finetuning_strategy import (
    apply_fine_tuning_strategy,
)


@dataclass
class FineTuningResult:

    strategy_name: str
    best_epoch: int
    best_accuracy: float
    final_accuracy: float
    training_time: float




def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct = 0

   

    for inputs, targets in dataloader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        optimizer.step()

        batch_size = inputs.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        correct
        / total_samples
    )

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    for inputs, targets in dataloader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        batch_size = inputs.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        correct
        / total_samples
    )

    return average_loss, accuracy

def run_fine_tuning(
    model,
    train_loader,
    val_loader,
    strategy,
    device,
    epochs=5,
    learning_rate=1e-3,
):
    apply_fine_tuning_strategy(
        model,
        strategy,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_epoch = 0
    final_accuracy = 0.0

    start_time = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        final_accuracy = val_accuracy

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_epoch = epoch

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Strategy={strategy.name} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train Acc={train_accuracy:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Acc={val_accuracy:.4f}"
        )

    training_time = (
        time.perf_counter()
        - start_time
    )

    return FineTuningResult(
        strategy_name=strategy.name,
        best_epoch=best_epoch,
        best_accuracy=best_accuracy,
        final_accuracy=final_accuracy,
        training_time=training_time,
    )