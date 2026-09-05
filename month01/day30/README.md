# Day 30 — ML Experiment Pipeline Capstone

## Overview

Day 30 integrates the engineering components developed during the previous days into a complete machine learning experiment pipeline.

The goal is not to implement another CNN from scratch, but to demonstrate how reusable ML components can be composed into a structured, reproducible, and testable experiment system.

The pipeline currently uses:

* CIFAR-10
* ResNet-18
* PyTorch
* Config-driven experiment setup
* Experiment tracking
* Checkpoint management
* Reproducibility control
* Automated testing

---

## Architecture

The Day 30 pipeline follows an adapter-based architecture.

```text
                         PipelineConfig
                               │
                               ▼
                      Experiment Factory
                               │
                               ▼
                      ExperimentTracker
                               │
                               ▼
                         Pipeline
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Data Adapter      Model Adapter   Experiment Adapter
              │                │                │
              ▼                ▼                │
          CIFAR-10          ResNet-18           │
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                       Training Adapter
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               Validation            Checkpoint
                    │                     │
                    └──────────┬──────────┘
                               ▼
                       Best Model Reload
                               │
                               ▼
                         Final Test
                               │
                               ▼
                      Experiment JSON
```

---

## Design Principles

### 1. Configuration-driven pipeline

The experiment is controlled through `PipelineConfig`.

```python
config = PipelineConfig(...)
```

Configuration is separated into:

* `DataConfig`
* `ModelConfig`
* `TrainingConfig`
* `ExperimentConfig`
* `CheckpointConfig`

This avoids scattering hyperparameters throughout the implementation.

---

### 2. Adapter pattern

Day 30 does not duplicate the implementations developed during Day 25.

Instead, adapters connect the new pipeline architecture to existing components.

```text
Day 30 Adapter
      │
      ▼
Day 25 validated implementation
```

For example:

```text
data_adapter
      ↓
Day 25 CIFAR-10 loader

model_adapter
      ↓
Day 25 ResNet18

training_adapter
      ↓
Day 25 training utilities
```

This reduces code duplication and preserves previously tested functionality.

---

### 3. Experiment Factory

The experiment tracker is created through the experiment factory.

```text
Experiment Factory
        ↓
Day 29 ExperimentTracker
```

The Day 30 pipeline receives the tracker instead of directly constructing it.

This keeps experiment creation separate from pipeline orchestration.

---

### 4. Experiment tracking

Each experiment records:

* Experiment ID
* Metadata
* Hyperparameters
* Training history
* Best epoch
* Best validation metric
* Best checkpoint
* Reproducibility metadata

Example:

```text
artifacts/
└── experiments/
    └── exp_YYYYMMDD_HHMMSS_xxxxxxxx.json
```

---

### 5. Best checkpoint management

During training, the pipeline tracks the best validation accuracy.

When a new best epoch is found:

```text
Validation Accuracy improves
          ↓
Save model checkpoint
          ↓
Record checkpoint metadata
```

The best checkpoint is then reloaded before final test evaluation.

```text
Best Checkpoint
      ↓
load_state_dict()
      ↓
Restored Model
      ↓
Final Test
```

---

### 6. Reproducibility

The experiment seed is part of the configuration.

```python
ExperimentConfig(
    seed=42,
    ...
)
```

The seed is passed into the experiment tracking and data-loading components.

The reproducibility test verifies that repeated runs with the same configuration and seed produce consistent experiment results.

---

## Project Structure

```text
day30/
│
├── configs/
│   └── pipeline_config.py
│
├── src/
│   ├── checkpoint_adapter.py
│   ├── data_adapter.py
│   ├── evaluation_adapter.py
│   ├── experiment_adapter.py
│   ├── experiment_factory.py
│   ├── model_adapter.py
│   ├── pipeline.py
│   └── training_adapter.py
│
├── tests/
│   ├── test_data_adapter.py
│   ├── test_model_adapter.py
│   ├── test_experiment_adapter.py
│   ├── test_training_adapter.py
│   ├── test_training_checkpoint_integration.py
│   ├── test_final_evaluation_integration.py
│   ├── test_pipeline.py
│   ├── test_capstone_artifacts.py
│   ├── test_checkpoint_recovery.py
│   └── test_reproducibility.py
│
├── run_pipeline.py
└── README.md
```

---

## Running the Pipeline

From the Day 30 directory:

```cmd
py run_pipeline.py
```

The current demonstration configuration uses a small CIFAR-10 subset to keep the end-to-end CPU execution practical.

The pipeline performs:

```text
1. Create experiment tracker
2. Initialize reproducibility
3. Create CIFAR-10 loaders
4. Create ResNet-18
5. Train model
6. Evaluate validation set
7. Track best epoch
8. Save best checkpoint
9. Reload best checkpoint
10. Evaluate test set
11. Save experiment record
```

---

## Testing

Run all Day 30 tests:

```cmd
py -m pytest tests -v
```

The test suite covers:

* Configuration and adapter behavior
* Data loading
* Model creation
* Experiment initialization
* Training
* Validation
* Checkpoint integration
* Final evaluation
* Pipeline orchestration
* Artifact consistency
* Checkpoint recovery
* Reproducibility

---

## Generated Artifacts

Successful pipeline execution produces two important artifacts.

### Experiment record

```text
artifacts/experiments/
```

Contains the experiment configuration, training history, best metric, checkpoint metadata, and reproducibility information.

### Model checkpoint

```text
artifacts/checkpoints/
```

Contains the best model state dictionary.

The experiment record references the corresponding checkpoint so that the trained model can be traced back to the exact experiment that produced it.

---

## Engineering Lessons

Day 30 demonstrates several important ML Engineering principles:

### Separation of concerns

Different components have clearly defined responsibilities.

### Reuse over duplication

Previously validated Day 25 components are reused through adapters.

### Configuration over hard-coding

Experiment parameters are represented explicitly through configuration objects.

### Reproducibility

Random seeds and experiment metadata are treated as first-class experiment information.

### Testability

The pipeline is decomposed into components that can be tested independently and together.

### Artifact traceability

Experiment records and model checkpoints are connected through experiment metadata.

---

## Day 30 Outcome

The result is a complete, testable experiment pipeline rather than an isolated training script.

The architecture provides a foundation for future extensions such as:

* Multiple model architectures
* Multiple datasets
* Hyperparameter experiments
* Experiment comparison
* Advanced metrics
* GPU training
* Distributed training
* ML experiment dashboards
* Model deployment

Day 30 completes the first major ML Engineering pipeline milestone of the 1 Year AI Engineer transformation plan.

