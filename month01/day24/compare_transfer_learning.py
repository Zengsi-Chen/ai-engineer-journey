import time

import torch
import json
from pathlib import Path
from dataclasses import asdict

from dataset import create_dataloaders
from experiments import (
    count_trainable_parameters,
    create_feature_extraction_experiment,
    create_fine_tuning_experiment,
    run_experiment,
)
from config import TransferLearningConfig
from reproducibility import set_seed



def main():
    config = TransferLearningConfig()

    set_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, test_loader = create_dataloaders(
        batch_size=config.batch_size,
        num_workers=0,
    )

    experiments = [
        (
            "Feature Extraction",
            create_feature_extraction_experiment,
        ),
        (
            "Fine-Tuning",
            create_fine_tuning_experiment,
        ),
    ]

    output = {
        "config": asdict(config),
        "results": {},
    }

    for name, create_experiment in experiments:

        set_seed(config.seed)

        model, optimizer = create_experiment(config)

        trainable_params = count_trainable_parameters(
            model
        )

        start_time = time.perf_counter()

        result = run_experiment(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=config.epochs,
        )

        elapsed_time = time.perf_counter() - start_time

        output["results"][name] = {
            "trainable_parameters": trainable_params,
            "best_test_accuracy": result[
                "best_test_accuracy"
            ],
            "best_epoch": result["best_epoch"],
            "training_time": elapsed_time,
        }

    results_path = Path(
        "artifacts/transfer_learning_results.json"
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        "\nResults saved to:",
        results_path,
    )

if __name__ == "__main__":
    main()