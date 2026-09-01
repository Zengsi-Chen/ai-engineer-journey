import torch
import json
from pathlib import Path

from model import ResNet18
from domain_experiment import (
    run_domain_shift_experiments,
)

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

from strategy_comparison import (
    StrategyResult,
    compare_strategies,
    find_best_strategy,
    find_most_efficient_strategy,
)

CHECKPOINT_PATH = (
    "artifacts/checkpoints/best_resnet18.pth"
    )

DATA_ROOT = "data"

BATCH_SIZE = 128

EPOCHS = 2

LEARNING_RATE = 1e-3


def save_result(result):

    results_dir = Path(
        "artifacts/results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        results_dir
        / f"{result.strategy_name}.json"
    )

    data = {
        "strategy": result.strategy_name,
        "best_epoch": result.best_epoch,
        "best_accuracy": result.best_accuracy,
        "final_accuracy": result.final_accuracy,
        "training_time": result.training_time,
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def get_strategy_results():

    return [

        StrategyResult(
            name="feature_extraction",
            best_accuracy=0.0000,
            best_epoch=0,
            training_time=0.0,
        ),

        StrategyResult(
            name="partial_fine_tuning",
            best_accuracy=0.6709,
            best_epoch=4,
            training_time=62937.49,
        ),

        StrategyResult(
            name="full_fine_tuning",
            best_accuracy=0.0000,
            best_epoch=0,
            training_time=0.0,
        ),

    ]


def load_model(device):
# Load the Day 25 ResNet-18 checkpoint.

    model = ResNet18(
        num_classes=10
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    return model

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # Target domain configuration
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

    strategies = (
        create_fine_tuning_strategies()
    )

    feature_extraction = strategies[0]

    partial_fine_tuning = strategies[1]

    full_fine_tuning = strategies[2]

    strategy = feature_extraction

    print(
        f"\nStrategy: {strategy.name}"
    )

    model = load_model(
        device=device
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
        learning_rate=LEARNING_RATE,
    )

    save_result(result)

    print("\nResult")

    print(
        f"Strategy: "
        f"{result.strategy_name}"
    )

    print(
        f"Best Epoch: "
        f"{result.best_epoch}"
    )

    print(
        f"Best Accuracy: "
        f"{result.best_accuracy:.4f}"
    )

    print(
        f"Final Accuracy: "
        f"{result.final_accuracy:.4f}"
    )

    print(
        f"Training Time: "
        f"{result.training_time:.2f} seconds"
    )

    baseline_accuracy = 0.3205

    results = get_strategy_results()

    comparison = compare_strategies(
        baseline_accuracy=baseline_accuracy,
        results=results,
    )

    best_strategy = find_best_strategy(
        results
    )

    most_efficient_strategy = (
        find_most_efficient_strategy(
            comparison
        )
    )

    print("\nStrategy Comparison")

    for item in comparison:

        print(
            f"\nStrategy: "
            f"{item['strategy']}"
        )

        print(
            f"Best Accuracy: "
            f"{item['best_accuracy']:.4f}"
        )

        print(
            f"Accuracy Gain: "
            f"{item['accuracy_gain']:+.4f}"
        )

        print(
            f"Best Epoch: "
            f"{item['best_epoch']}"
        )

        print(
            f"Training Time: "
            f"{item['training_time']:.2f} seconds"
        )

        print(
            f"\nBest Strategy: "
            f"{best_strategy.name}"
        )

        print(
            f"Most Efficient Strategy: "
            f"{most_efficient_strategy['strategy']}"
        )

        print(
            f"Most Efficient Strategy: "
            f"{most_efficient_strategy['strategy']}"
        )

if __name__ == "__main__":
    main()