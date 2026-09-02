import copy
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from optimizer_factory import (
    create_optimizer,
    )

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


def calculate_accuracy(
    model,
    data_loader,
    device,
    ):
    """
    Calculate classification accuracy.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, targets in data_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

    if total == 0:
        return 0.0

    return correct / total


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    ):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in train_loader:

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

        batch_size = targets.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        return 0.0

    return (
        total_loss
        / total_samples
    )
    

def run_fine_tuning(
    model,
    train_loader,
    val_loader,
    strategy,
    device,
    epochs,
    config,
    ):
    """
    Run one complete fine-tuning experiment.

    ```
    The strategy controls which layers are trainable.

    The config controls the optimizer and
    learning-rate configuration.
    """

    # --------------------------------------------------
    # Apply Fine-Tuning Strategy
    # --------------------------------------------------

    apply_fine_tuning_strategy(
        model=model,
        strategy=strategy,
    )

    # --------------------------------------------------
    # Create Optimizer
    # --------------------------------------------------

    optimizer = create_optimizer(
        model=model,
        config=config,
    )
    print("\nOptimizer Parameter Groups:")

    for index, group in enumerate(
        optimizer.param_groups,
    ):

        parameter_count = sum(
            parameter.numel()
            for parameter in group["params"]
        )

        print(
            f"Group {index}: "
            f"lr={group['lr']:.1e}, "
            f"parameters={parameter_count:,}"
        )
        
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Experiment State
    # --------------------------------------------------

    best_accuracy = 0.0

    best_epoch = 0

    best_model_state = None

    # --------------------------------------------------
    # Training Timer
    # --------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_accuracy = calculate_accuracy(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{epochs:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save Best Model
        # --------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

    # --------------------------------------------------
    # Training Time
    # --------------------------------------------------

    training_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # Restore Best Model
    # --------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    # --------------------------------------------------
    # Final Accuracy
    # --------------------------------------------------

    final_accuracy = calculate_accuracy(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    # --------------------------------------------------
    # Return Experiment Result
    # --------------------------------------------------

    return FineTuningResult(
        strategy_name=strategy.name,
        best_epoch=best_epoch,
        best_accuracy=best_accuracy,
        final_accuracy=final_accuracy,
        training_time=training_time,
    )

