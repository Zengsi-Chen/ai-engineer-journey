# Day 29 — Experiment Tracking & Model Comparison

## Overview

Machine learning development is not just about training a model and saving its final accuracy.

A real experiment should answer questions such as:

* Which hyperparameters produced this result?
* What happened during training?
* At which epoch was the best result achieved?
* Which checkpoint produced the best metric?
* Which experiment performed best among multiple runs?
* Does the checkpoint file actually exist?
* Can the experiment be reproduced later?

This project builds a lightweight experiment management system for tracking, persisting, comparing, and selecting machine learning experiments.

---

# Learning Objectives

This project covers:

1. Why saving only accuracy is not enough
2. Experiment IDs
3. Hyperparameter tracking
4. Training metrics tracking
5. Best checkpoint tracking
6. Experiment metadata
7. JSON experiment records
8. Multi-experiment comparison
9. Best model selection
10. Experiment reproducibility
11. Moving from manual experiments to automated experiment management

---

# Project Architecture

```text
Experiment
    │
    ├── Experiment ID
    │
    ├── Metadata
    │
    ├── Hyperparameters
    │
    ├── Reproducibility
    │       ├── Random Seed
    │       └── Environment Metadata
    │
    ├── Training History
    │
    ├── Best Metric
    │
    └── Best Checkpoint
            │
            ▼
     Experiment Record
            │
            ▼
        JSON File
            │
            ▼
     Experiment Comparator
            │
            ├── Load Experiments
            ├── Compare Experiments
            ├── Best Experiment
            │
            ▼
     Checkpoint Validation
            │
            ▼
        Best Model
```

---

# Project Structure

```text
day29/
│
├── artifacts/
│   └── experiments/
│       └── .gitkeep
│
├── src/
│   ├── experiment_tracker.py
│   ├── experiment_comparator.py
│   └── reproducibility.py
│
├── tests/
│   ├── test_experiment_tracker.py
│   ├── test_experiment_comparator.py
│   └── test_reproducibility.py
│
├── .gitignore
└── README.md
```

---

# 1. Experiment Tracking

The `ExperimentTracker` records the complete lifecycle of an experiment.

An experiment record includes:

```python
{
    "experiment_id": "...",
    "metadata": {...},
    "hyperparameters": {...},
    "training_history": [...],
    "best_epoch": ...,
    "best_metric": {...},
    "best_checkpoint": {...},
    "reproducibility": {...}
}
```

Instead of manually writing down a final result such as:

```text
Accuracy = 0.94
```

the experiment system records the context required to understand and reproduce the result.

---

# 2. Experiment ID

Each experiment receives a unique identifier.

Example:

```text
exp_20260903_222759_e5c9be22
```

This allows each experiment record to be uniquely identified and linked to its metrics, hyperparameters, and checkpoint.

---

# 3. Hyperparameter Tracking

Hyperparameters can be recorded with the experiment.

Example:

```python
{
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 20,
}
```

This makes it possible to answer:

> Which configuration produced the best result?

---

# 4. Training Metrics Tracking

A single final metric does not describe the complete training process.

The system stores training history across epochs.

Example:

```python
{
    "epoch": 5,
    "train_loss": 0.42,
    "val_accuracy": 0.88,
}
```

This enables later analysis of:

* Training progress
* Validation performance
* Best epoch
* Performance trends
* Experiment comparison

---

# 5. Best Metric and Best Epoch

The best result is tracked separately from the full training history.

Example:

```python
{
    "name": "val_accuracy",
    "value": 0.94
}
```

The corresponding epoch is also recorded:

```text
Best Epoch = 12
```

This prevents the common mistake of assuming the final epoch is always the best epoch.

---

# 6. Best Checkpoint Tracking

The best model is not just the highest score.

The system records the model artifact associated with the best result.

Example:

```python
{
    "path": "artifacts/checkpoints/best_model.pt",
    "epoch": 12,
    "metric_name": "val_accuracy",
    "metric_value": 0.94
}
```

The checkpoint is linked directly to the experiment record.

---

# 7. JSON Experiment Records

Experiments are persisted as JSON files.

Example:

```text
artifacts/
└── experiments/
    └── exp_20260903_222759_e5c9be22.json
```

JSON persistence allows experiments to survive after the Python process ends.

This enables:

```text
Run Experiment
      ↓
Save JSON Record
      ↓
Restart Python
      ↓
Load Experiment
      ↓
Compare Results
```

---

# 8. Multi-Experiment Comparison

The `ExperimentComparator` loads multiple experiment records automatically.

Conceptually:

```text
Experiment A
Accuracy = 0.85

Experiment B
Accuracy = 0.91

Experiment C
Accuracy = 0.88
```

The comparator can identify:

```text
Best Experiment
      ↓
Experiment B
      ↓
Accuracy = 0.91
```

---

# 9. Best Model Selection

The highest-scoring experiment is not automatically a usable model.

The system validates:

1. The best experiment has checkpoint metadata
2. Required checkpoint fields are present
3. The checkpoint metric matches the experiment metric
4. The checkpoint path exists
5. The checkpoint path is a file

The final workflow is:

```text
Best Experiment
      ↓
Best Checkpoint Metadata
      ↓
Metadata Validation
      ↓
Checkpoint File Validation
      ↓
Best Model
```

This provides artifact consistency between experiment records and actual model files.

---

# 10. Experiment Reproducibility

Machine learning results can vary because of random initialization, data shuffling, augmentation, and hardware or framework differences.

The project records reproducibility information such as:

```text
Random Seed
Python Version
PyTorch Version
Platform
Device
CUDA Version
cuDNN Version
```

Example:

```python
{
    "seed": 42,
    "environment": {
        "python_version": "...",
        "pytorch_version": "...",
        "platform": "...",
        "device": "cpu",
        "cuda_version": None,
        "cudnn_version": None,
    }
}
```

The project also applies the seed to:

* Python `random`
* NumPy
* PyTorch CPU
* PyTorch CUDA when available

---

# 11. From Manual Experiments to Automated Experiment Management

A manual workflow often looks like:

```text
Run Training
      ↓
Check Accuracy
      ↓
Write Result Somewhere
      ↓
Run Another Experiment
      ↓
Compare Results Manually
```

This project transforms the workflow into:

```text
Run Experiment
      ↓
Automatic Tracking
      ↓
Record Hyperparameters
      ↓
Record Training Metrics
      ↓
Record Best Checkpoint
      ↓
Record Reproducibility Metadata
      ↓
Save JSON Record
      ↓
Automatically Load Experiments
      ↓
Compare Experiments
      ↓
Validate Best Model
```

---

# Running Tests

From the `day29` directory:

```cmd
py -m pytest tests -v
```

Run individual test modules:

```cmd
py -m pytest tests\test_experiment_tracker.py -v
```

```cmd
py -m pytest tests\test_experiment_comparator.py -v
```

```cmd
py -m pytest tests\test_reproducibility.py -v
```

---

# Engineering Concepts Practiced

This project practices:

* Experiment tracking
* Persistent experiment records
* JSON serialization
* Metadata management
* Hyperparameter tracking
* Training history tracking
* Model checkpoint tracking
* Artifact validation
* Multi-experiment comparison
* Best model selection
* Reproducibility
* Random seed management
* Environment metadata
* Automated testing
* Fail-fast validation

---

# Key Engineering Principle

The most important idea from this project is:

```text
Best Score
≠
Best Model
```

A production-ready model selection process requires:

```text
Best Score
+
Experiment Context
+
Valid Checkpoint Metadata
+
Existing Model Artifact
+
Reproducibility Information
```

---

# Next Step

The next stage is to connect experiment tracking with actual training pipelines so that experiments are recorded automatically during real CNN and transfer learning workflows.
