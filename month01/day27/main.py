import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import ResNet18

from domain_shift import (
    DomainShiftConfig,
)

from target_domain_data import (
    create_target_domain_data,
)

from finetuning_pipeline import (
    run_fine_tuning,
)

from finetuning_strategy import (
    create_fine_tuning_strategies,
)

from finetuning_config import (
    FineTuningConfig,
)

from experiment_report import (
    generate_experiment_report,
)
# ==================================================

# Paths

# ==================================================

CHECKPOINT_PATH = (
    "artifacts/checkpoints/best_resnet18.pth"
)

DATA_ROOT = "data"

RESULTS_PATH = (
    "artifacts/results/day27_results.json"
)

REPORT_PATH = (
    "artifacts/reports/"
    "day27_experiment_report.md"
)

# ==================================================

# Experiment Settings

# ==================================================

BATCH_SIZE = 128

EPOCHS = 2

def load_model(device):
    """
    Load the same checkpoint for each experiment.
    """


    model = ResNet18(
        num_classes=10
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    return model
    

def run_experiment(
    name,
    target_data,
    strategy,
    device,
    config,
    ):
    """
    Run one experiment from the same
    initial checkpoint.
    """

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Experiment: {name}"
    )

    print(
        f"{'=' * 60}"
    )

    # Every experiment starts from
    # the same Day 25 checkpoint.
    model = load_model(
        device=device
    )

    experiment_start_time = (
        time.perf_counter()
    )

    result = run_fine_tuning(
        model=model,
        train_loader=(
            target_data.train_loader
        ),
        val_loader=(
            target_data.val_loader
        ),
        strategy=strategy,
        device=device,
        epochs=EPOCHS,
        config=config,
    )

    experiment_total_time = (
        time.perf_counter()
        - experiment_start_time
    )

    print(
        f"\nTotal Experiment Time: "
        f"{experiment_total_time:.2f} seconds"
    )

    return result, experiment_total_time


def print_result(
    name,
    result,
    experiment_time,
    ):
    """
    Print experiment results.
    """

    print(
        f"\n{name}"
    )

    print(
        f"Best Accuracy: "
        f"{result.best_accuracy:.4f}"
    )

    print(
        f"Best Epoch: "
        f"{result.best_epoch}"
    )

    print(
        f"Final Accuracy: "
        f"{result.final_accuracy:.4f}"
    )

    print(
        f"Training Time: "
        f"{result.training_time:.2f} seconds"
    )

    print(
        f"Total Experiment Time: "
        f"{experiment_time:.2f} seconds"
    )


def save_results(
    results,
    ):
    """
    Save experiment results to JSON.
    """

    
    results_path = Path(
        RESULTS_PATH
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_results = {}

    for name, data in results.items():

        result = data["result"]

        experiment_time = (
            data["experiment_time"]
        )

        serializable_results[
            name
        ] = {
            **asdict(result),
            "total_experiment_time": (
                experiment_time
            ),
        }

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable_results,
            file,
            indent=4,
        )

    print(
        f"\nResults saved to:"
    )

    print(
        results_path
    )

def print_tradeoff_analysis(
    results,
):
    """
    Compare accuracy and training-time
    trade-offs against Uniform LR.
    """

    baseline = results[
        "uniform_lr"
    ]["result"]

    baseline_accuracy = (
        baseline.best_accuracy
    )

    baseline_time = (
        baseline.training_time
    )

    print(
        f"\n{'=' * 80}"
    )

    print(
        "Accuracy / Training Time Trade-off"
    )

    print(
        f"{'=' * 80}"
    )

    print(
        f"{'Strategy':<35}"
        f"{'Accuracy':>12}"
        f"{'Δ Accuracy':>14}"
        f"{'Time (s)':>14}"
        f"{'Δ Time':>14}"
    )

    print(
        "-" * 80
    )

    for name, data in results.items():

        result = data["result"]

        accuracy = (
            result.best_accuracy
        )

        training_time = (
            result.training_time
        )

        accuracy_difference = (
            accuracy
            - baseline_accuracy
        )

        time_difference = (
            training_time
            - baseline_time
        )

        print(
            f"{name:<35}"
            f"{accuracy:>12.4f}"
            f"{accuracy_difference:>+14.4f}"
            f"{training_time:>14.2f}"
            f"{time_difference:>+14.2f}"
        )   

def main():

    # --------------------------------------------------
    # Run Experiments
    # --------------------------------------------------

    results = {}

    # --------------------------------------------------
    # Total Program Timer
    # --------------------------------------------------

    program_start_time = (
        time.perf_counter()
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )
    # --------------------------------------------------
    # Total Program Timer
    # --------------------------------------------------

    program_start_time = (
        time.perf_counter()
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------
    # Target Domain
    # --------------------------------------------------

    target_config = DomainShiftConfig(
        brightness=0.2,
        contrast=0.2,
        grayscale=False,
        noise_std=0.02,
    )

    target_data = (
        create_target_domain_data(
            root=DATA_ROOT,
            name="moderate_shift",
            config=target_config,
            batch_size=BATCH_SIZE,
        )
    )

    # --------------------------------------------------
    # Fine-Tuning Strategy
    # --------------------------------------------------

    strategies = (
        create_fine_tuning_strategies()
    )

    strategy = next(
        strategy
        for strategy in strategies
        if (
            strategy.name
            == "partial_fine_tuning"
        )
    )

    print(
        f"\nFine-Tuning Strategy: "
        f"{strategy.name}"
    )

    # --------------------------------------------------
    # Experiment A
    # Uniform Learning Rate
    # --------------------------------------------------

    uniform_config = FineTuningConfig(
        base_learning_rate=1e-3,
        layer_wise_decay=1.0,
        weight_decay=1e-4,
    )

    (
        uniform_result,
        uniform_experiment_time,
    ) = run_experiment(
        name="Uniform LR",
        target_data=target_data,
        strategy=strategy,
        device=device,
        config=uniform_config,
    )

    # --------------------------------------------------
    # Experiment B
    # Manual Discriminative Learning Rates
    # --------------------------------------------------

    manual_discriminative_config = (
        FineTuningConfig(
            weight_decay=1e-4,
            discriminative_learning_rates={
                "layer3": 1e-4,
                "layer4": 3e-4,
                "classifier": 1e-3,
            },
        )
    )

    (
        manual_result,
        manual_experiment_time,
    ) = run_experiment(
        name="Manual Discriminative LR",
        target_data=target_data,
        strategy=strategy,
        device=device,
        config=(
            manual_discriminative_config
        ),
    )

    # --------------------------------------------------
    # Experiment C
    # Automatic Layer-wise Decay
    # --------------------------------------------------

    automatic_layer_wise_config = (
        FineTuningConfig(
            base_learning_rate=1e-3,
            layer_wise_decay=0.3,
            weight_decay=1e-4,
        )
    )

    (
        automatic_result,
        automatic_experiment_time,
    ) = run_experiment(
        name="Automatic Layer-wise Decay",
        target_data=target_data,
        strategy=strategy,
        device=device,
        config=(
            automatic_layer_wise_config
        ),
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results = {
        "uniform_lr": {
            "result": uniform_result,
            "experiment_time": (
                uniform_experiment_time
            ),
        },
        "manual_discriminative_lr": {
            "result": manual_result,
            "experiment_time": (
                manual_experiment_time
            ),
        },
        "automatic_layer_wise_decay": {
            "result": automatic_result,
            "experiment_time": (
                automatic_experiment_time
            ),
        },
    }

    print(
        f"\n{'=' * 60}"
    )

    print(
        "Day 27 Experiment Results"
    )

    print(
        f"{'=' * 60}"
    )

    print_result(
        "Uniform LR",
        uniform_result,
        uniform_experiment_time,
    )

    print_result(
        "Manual Discriminative LR",
        manual_result,
        manual_experiment_time,
    )

    print_result(
        "Automatic Layer-wise Decay",
        automatic_result,
        automatic_experiment_time,
    )

    # --------------------------------------------------
    # Accuracy Ranking
    # --------------------------------------------------

    ranked_results = sorted(
        results.items(),
        key=lambda item: (
            item[1]["result"].best_accuracy
        ),
        reverse=True,
    )

    print(
        f"\n{'=' * 60}"
    )

    print(
        "Accuracy Ranking"
    )

    print(
        f"{'=' * 60}"
    )

    for rank, (
        name,
        data,
    ) in enumerate(
        ranked_results,
        start=1,
    ):

        result = data["result"]

        print(
            f"{rank}. "
            f"{name}: "
            f"{result.best_accuracy:.4f}"
        )

    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------

    print_tradeoff_analysis(
        results
    )

    generate_experiment_report(
        results=results,
        report_path=REPORT_PATH,
    )

    save_results(
        results
    )

    # --------------------------------------------------
    # Total Program Time
    # --------------------------------------------------

    program_total_time = (
        time.perf_counter()
        - program_start_time
    )

    print(
        f"\nTotal Program Time: "
        f"{program_total_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
