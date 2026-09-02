# Day 27 — Layer-wise Learning Rates for Fine-Tuning

## Overview

Day 27 focuses on learning-rate strategies for fine-tuning a pretrained ResNet-18 model.

The main goal is to compare different learning-rate strategies under controlled experimental conditions.

The three strategies are:

1. Uniform Learning Rate
2. Manual Discriminative Learning Rates
3. Automatic Layer-wise Learning-rate Decay

The experiments use the same:

* pretrained checkpoint
* target domain
* fine-tuning strategy
* training budget
* batch size
* optimizer settings
* weight decay

Only the learning-rate strategy changes.

This makes the comparison a controlled experiment.

---

# Learning Objectives

By completing Day 27, I learned how to:

* implement layer-wise learning rates
* create optimizer parameter groups
* implement discriminative learning rates
* automatically assign learning rates based on layer depth
* combine fine-tuning strategies with learning-rate strategies
* design controlled machine-learning experiments
* record training time
* compare accuracy and computational cost
* save experiment results as JSON
* generate Markdown experiment reports automatically
* write unit tests for optimizer logic
* separate experiment orchestration from model-training logic

---

# Learning-Rate Strategies

## 1. Uniform Learning Rate

All trainable layers use the same learning rate.

```text
Layer 3      → 0.001
Layer 4      → 0.001
Classifier   → 0.001
```

This is the simplest baseline.

```text
Uniform LR
```

Advantages:

* simple
* easy to configure
* easy to reproduce

Limitations:

* different layers may require different update speeds
* pretrained layers and task-specific layers are treated identically

---

## 2. Manual Discriminative Learning Rates

Different layers use manually selected learning rates.

Example:

```text
Layer 3      → 0.0001
Layer 4      → 0.0003
Classifier   → 0.0010
```

The intuition is:

```text
Earlier layers
    ↓
More general pretrained features
    ↓
Smaller updates


Later layers
    ↓
More task-specific features
    ↓
Larger updates
```

---

## 3. Automatic Layer-wise Learning-rate Decay

Instead of manually assigning every learning rate, the learning rates are generated automatically.

Conceptually:

```text
LR(layer)
=
base_lr × decay^depth
```

For example:

```text
Classifier   → highest LR
Layer 4      → lower LR
Layer 3      → lower LR
```

This provides a scalable way to implement discriminative learning rates.

---

# Fine-Tuning Strategy

The Day 27 experiments use:

```text
partial_fine_tuning
```

The trainable layers are:

```text
layer3
layer4
classifier
```

Earlier layers remain frozen.

This reduces unnecessary updates to general pretrained features.

---

# Optimizer Parameter Groups

Different learning rates are implemented using optimizer parameter groups.

Conceptually:

```python
optimizer = torch.optim.Adam(
    [
        {
            "params": layer3.parameters(),
            "lr": 1e-4,
        },
        {
            "params": layer4.parameters(),
            "lr": 3e-4,
        },
        {
            "params": classifier.parameters(),
            "lr": 1e-3,
        },
    ]
)
```

This allows each trainable part of the network to use a different learning rate.

---

# Experiment Design

The three experiments use the same:

```text
✓ Model checkpoint
✓ Target-domain dataset
✓ Fine-tuning strategy
✓ Trainable layers
✓ Batch size
✓ Number of epochs
✓ Optimizer
✓ Weight decay
```

The only experimental variable is:

```text
Learning-rate strategy
```

Therefore:

```text
Uniform LR
        vs
Manual Discriminative LR
        vs
Automatic Layer-wise Decay
```

is a controlled comparison.

---

# Project Structure

```text
day27/
│
├── main.py
├── model.py
├── resnet.py
│
├── domain_shift.py
├── target_domain_data.py
│
├── finetuning_config.py
├── finetuning_strategy.py
├── finetuning_pipeline.py
├── optimizer_factory.py
├── experiment_report.py
│
├── artifacts/
│   ├── checkpoints/
│   │   └── best_resnet18.pth
│   │
│   ├── results/
│   │   └── day27_results.json
│   │
│   └── reports/
│       └── day27_experiment_report.md
│
└── tests/
    ├── test_optimizer_factory.py
    └── test_experiment_report.py
```

---

# Running the Experiments

Run:

```cmd
py main.py
```

The program will:

```text
Load pretrained checkpoint
        ↓
Create target-domain data
        ↓
Select fine-tuning strategy
        ↓
Run Uniform LR experiment
        ↓
Run Manual Discriminative LR experiment
        ↓
Run Automatic Layer-wise Decay experiment
        ↓
Compare results
        ↓
Record training time
        ↓
Save JSON results
        ↓
Generate Markdown report
```

---

# Results

The experiment compares:

| Strategy                   |            Best Accuracy |  Best Epoch | Training Time |
| -------------------------- | -----------------------: | ----------: | ------------: |
| Uniform LR                 | 0.5766                   | 2           |   13308.13    |
| Manual Discriminative LR   | 0.5924                   | 2           |   53010.78    |
| Automatic Layer-wise Decay | 0.5893                   | 2           |   7266.17     |

The exact experimental results are automatically saved to:

```text
artifacts/results/day27_results.json
```

A Markdown report is automatically generated at:

```text
artifacts/reports/day27_experiment_report.md
```

---

# Accuracy and Training-Time Trade-off

Accuracy alone is not enough to evaluate an experiment.

The following factors should be considered:

```text
Best Accuracy
        +
Training Time
        +
Accuracy Improvement
        +
Computational Cost
```

For example:

```text
Strategy A

Accuracy: 0.7500
Time:     100 seconds
```

```text
Strategy B

Accuracy: 0.7510
Time:     180 seconds
```

Although Strategy B achieves slightly higher accuracy, the additional computational cost may not justify the small improvement.

Therefore:

```text
Best Accuracy
```

does not always mean:

```text
Best Engineering Choice
```

---

# Experiment Result Persistence

Experiment results are saved as JSON.

Example:

```json
{
    "uniform_lr": {
        "strategy_name": "partial_fine_tuning",
        "best_epoch": 5,
        "best_accuracy": 0.0000,
        "final_accuracy": 0.0000,
        "training_time": 0.00
    }
}
```

This allows experiments to be analyzed after training instead of relying only on console output.

---

# Automatic Experiment Reports

The experiment report includes:

* experiment setup
* result table
* accuracy ranking
* accuracy differences
* training-time differences
* best experiment
* automatic conclusion

The report is generated automatically.

```text
Experiment
    ↓
JSON Results
    ↓
Markdown Report
```

---

# Unit Tests

The following components are tested:

```text
Layer-wise LR calculation
        ✓

Decay = 1.0 behavior
        ✓

Trainable layer detection
        ✓

Parameter group creation
        ✓

Uniform LR
        ✓

Manual discriminative LR
        ✓

Automatic layer-wise decay
        ✓

Manual LR priority
        ✓

Best experiment detection
        ✓

Trade-off calculation
        ✓

Markdown report generation
        ✓
```

Run the tests:

```cmd
py -m pytest tests\ -v
```

---

# Key Engineering Lessons

## 1. Fine-Tuning Is More Than Freezing Layers

Fine-tuning requires decisions about:

```text
Which layers to train?
        +
How fast should each layer learn?
```

---

## 2. Learning Rate Can Be Layer-Specific

Different layers represent different levels of abstraction.

```text
Early Layers
    ↓
General Features
    ↓
Smaller LR


Later Layers
    ↓
Task-Specific Features
    ↓
Larger LR
```

---

## 3. More Complex Strategies Do Not Guarantee Better Results

Automatic layer-wise learning-rate decay is more sophisticated than a uniform learning rate.

However:

```text
More Complex
≠
Better
```

The strategy must be evaluated experimentally.

---

## 4. Controlled Experiments Are Essential

A valid comparison requires:

```text
Change One Important Variable
        ↓
Keep Everything Else Constant
        ↓
Measure the Difference
```

For Day 27:

```text
Learning-rate strategy
```

is the main experimental variable.

---

## 5. Accuracy Is Not the Only Metric

A strategy should be evaluated using:

```text
Accuracy
+
Training Time
+
Computational Cost
+
Reproducibility
```

---

# Final Conclusion

Day 27 extends the fine-tuning system developed in previous days by introducing layer-specific learning-rate strategies.

Three approaches were implemented and compared:

```text
Uniform Learning Rate
        ↓
Manual Discriminative Learning Rates
        ↓
Automatic Layer-wise Learning-rate Decay
```

The project now supports:

```text
Fine-Tuning Strategy
        +
Layer-wise Learning Rates
        +
Optimizer Parameter Groups
        +
Controlled Experiments
        +
Training-Time Measurement
        +
JSON Result Persistence
        +
Automatic Markdown Reports
        +
Automated Unit Tests
```

The most important lesson from Day 27 is:

```text
A learning-rate strategy should not be chosen only because it is more sophisticated.

It should be evaluated under controlled experimental conditions using both model performance and computational cost.
```

---

# Next Step

Day 28 will build on the current fine-tuning system and continue the progression toward more advanced computer-vision and AI-engineering workflows.
