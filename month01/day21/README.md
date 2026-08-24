# Day 21 — Production-Style Classification Pipeline

## Overview

Day 21 focuses on engineering a reusable machine learning
training and evaluation pipeline.

The project separates:

- Configuration
- Dataset
- Model
- Loss
- Optimizer
- Scheduler
- Trainer
- Checkpoint
- Early Stopping
- Evaluation
- Metrics
- Threshold Optimization
- Experiment Tracking
- Artifact Management
- Pipeline Result
- Inference

## Features

### Training

- Configurable training epochs
- Optimizer factory
- Scheduler factory
- Early stopping
- Best model checkpoint
- Full training checkpoint

### Resume Training

The pipeline supports resuming an experiment from:

- model state
- optimizer state
- scheduler state
- early stopping state
- epoch

### Evaluation

The pipeline supports:

- Accuracy
- Precision
- Recall
- F1
- Threshold optimization

### Experiment Tracking

Each experiment receives a unique experiment ID.

Artifacts are stored under:

artifacts/experiments/<experiment_id>/

Example:

artifacts/
└── experiments/
    └── 20260825_010839/
        ├── checkpoints/
        │   ├── checkpoint.pth
        │   └── best_model.pth
        ├── plots/
        │   └── loss_curve.png
        └── experiment.json

## Run

From the Day 21 directory:

```cmd
py main.py