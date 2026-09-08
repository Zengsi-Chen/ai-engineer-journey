import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "day34_trainer_experiment.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "day34_training_report.md"
)


def main() -> None:
    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    history = data["history"]

    best = max(
        history,
        key=lambda item: item["val_dice"],
    )

    final = history[-1]

    report = f"""# Day 34 — Training Experiment Report

## 1. Experiment Overview

- Dataset: Kvasir-SEG
- Task: Binary Medical Image Segmentation
- Model: U-Net
- Model Features: {data["model"]["features"]}
- Image Size: {data["training"]["image_size"]}
- Batch Size: {data["training"]["batch_size"]}
- Device: {data["experiment"]["device"]}
- Seed: {data["experiment"]["seed"]}

## 2. Training Configuration

- Requested Epochs: {data["training"]["epochs_requested"]}
- Learning Rate: {data["training"]["learning_rate"]}
- Weight Decay: {data["training"]["weight_decay"]}
- Loss: BCE + Dice
- BCE Weight: 0.5
- Dice Weight: 0.5

## 3. Scheduler

- Scheduler: {data["scheduler"]["name"]}
- Monitor: {data["scheduler"]["monitor"]}
- Factor: {data["scheduler"]["factor"]}
- Patience: {data["scheduler"]["patience"]}
- Minimum LR: {data["scheduler"]["min_lr"]}

## 4. Early Stopping

- Monitor: {data["early_stopping"]["monitor"]}
- Patience: {data["early_stopping"]["patience"]}
- Minimum Delta: {data["early_stopping"]["min_delta"]}

## 5. Best Validation Result

- Best Epoch: {best["epoch"]}
- Best Validation Dice: {best["val_dice"]:.6f}
- Validation IoU at Best Epoch: {best["val_iou"]:.6f}
- Validation Loss at Best Epoch: {best["val_loss"]:.6f}

## 6. Final Epoch

- Final Epoch: {final["epoch"]}
- Final Validation Dice: {final["val_dice"]:.6f}
- Final Validation IoU: {final["val_iou"]:.6f}
- Final Validation Loss: {final["val_loss"]:.6f}

## 7. Training Behavior

The experiment records training loss, validation loss,
validation Dice, validation IoU, and learning rate for
every epoch.

The best checkpoint is selected using validation Dice.

ReduceLROnPlateau is used to reduce the learning rate
when validation Dice stops improving.

Early stopping terminates training when validation Dice
fails to improve beyond the configured patience and
minimum delta.

## 8. Engineering Interpretation

The experiment demonstrates an end-to-end training controller
with:

1. deterministic experiment setup,
2. configurable model creation,
3. reusable loss and optimizer components,
4. validation metrics,
5. best-model checkpointing,
6. learning-rate scheduling,
7. early stopping,
8. structured experiment history,
9. reproducible result artifacts.

## 9. Artifacts

- Experiment result:
  `artifacts/experiments/day34_trainer_experiment.json`

- Best checkpoint:
  `artifacts/checkpoints/best_model.pt`

- Loss curve:
  `artifacts/figures/day34_loss_curve.png`

- Segmentation metrics:
  `artifacts/figures/day34_segmentation_metrics.png`

- Learning-rate curve:
  `artifacts/figures/day34_learning_rate.png`
"""

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()