# Day 23 — Computer Vision: CIFAR-10 CNN Baseline

## Overview

Day 23 officially begins the Computer Vision stage of the AI Engineer Journey.

The goal of this day is to build a complete image classification pipeline using CIFAR-10, including:

- CIFAR-10 dataset
- Data augmentation
- CNN architecture
- Training pipeline
- Validation
- Best model checkpointing
- Test evaluation
- Confusion matrix
- Per-class accuracy
- Failure analysis
- Training visualization

The focus is not only on building a CNN, but on applying the ML engineering practices developed in previous days to a real Computer Vision task.

---

## Learning Pipeline

```text
CIFAR-10
    ↓
Data Augmentation
    ↓
DataLoader
    ↓
Baseline CNN
    ↓
CrossEntropyLoss
    ↓
Adam Optimizer
    ↓
Training
    ↓
Validation
    ↓
Best Model Checkpoint
    ↓
Test Set
    ↓
Confusion Matrix
    ↓
Failure Analysis