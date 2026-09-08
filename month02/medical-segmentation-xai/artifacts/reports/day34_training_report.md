# Day 34 — Training Experiment Report

## 1. Experiment Overview

- Dataset: Kvasir-SEG
- Task: Binary Medical Image Segmentation
- Model: U-Net
- Model Features: [16, 32, 64, 128]
- Image Size: 256
- Batch Size: 4
- Device: cpu
- Seed: 42

## 2. Training Configuration

- Requested Epochs: 10
- Learning Rate: 0.001
- Weight Decay: 0.0001
- Loss: BCE + Dice
- BCE Weight: 0.5
- Dice Weight: 0.5

## 3. Scheduler

- Scheduler: plateau
- Monitor: val_dice
- Factor: 0.5
- Patience: 2
- Minimum LR: 1e-06

## 4. Early Stopping

- Monitor: val_dice
- Patience: 3
- Minimum Delta: 0.001

## 5. Best Validation Result

- Best Epoch: 10
- Best Validation Dice: 0.636565
- Validation IoU at Best Epoch: 0.499315
- Validation Loss at Best Epoch: 0.372667

## 6. Final Epoch

- Final Epoch: 10
- Final Validation Dice: 0.636565
- Final Validation IoU: 0.499315
- Final Validation Loss: 0.372667

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
