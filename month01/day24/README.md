# Day 24 — Transfer Learning

## Overview

Day 24 focuses on **Transfer Learning** for computer vision using a pretrained ResNet18 model on CIFAR-10.

The goal is not only to train a pretrained CNN, but to build a reproducible experiment pipeline that compares:

1. Feature Extraction
2. Partial Fine-Tuning

The experiment evaluates the trade-off between model performance and training cost.

---

## Learning Objectives

* Understand Transfer Learning
* Use ImageNet-pretrained ResNet18
* Freeze pretrained layers
* Replace the pretrained classifier
* Perform partial Fine-Tuning
* Use Differential Learning Rates
* Save and reload the best model
* Build reproducible experiments
* Compare different training strategies
* Record experiment results

---

## Transfer Learning Strategies

### Feature Extraction

Only the final classifier is trained.

```text
ResNet18
│
├── Layer 1   Frozen
├── Layer 2   Frozen
├── Layer 3   Frozen
├── Layer 4   Frozen
└── FC        Trainable
```

This strategy has very few trainable parameters and is computationally efficient.

---

### Partial Fine-Tuning

The final ResNet block and classifier are trained.

```text
ResNet18
│
├── Layer 1   Frozen
├── Layer 2   Frozen
├── Layer 3   Frozen
├── Layer 4   Trainable
└── FC        Trainable
```

Layer4 uses a smaller learning rate than the newly initialized classifier.

```text
Layer4 → 1e-4
FC     → 1e-3
```

This allows the pretrained high-level features to adapt gradually to CIFAR-10.

---

## Dataset

**CIFAR-10**

* 10 classes
* 32×32 RGB images
* 50,000 training images
* 10,000 test images

The model uses ImageNet-compatible preprocessing for the pretrained ResNet18 backbone.

---

## Model

```text
ResNet18
↓
ImageNet pretrained weights
↓
Replace final FC layer
↓
10-class classifier
```

---

## Experiments

Two controlled experiments were performed.

### Experiment A — Feature Extraction

* Pretrained backbone frozen
* Only classifier trained
* Classifier learning rate: `1e-3`

### Experiment B — Fine-Tuning

* Layer1 frozen
* Layer2 frozen
* Layer3 frozen
* Layer4 trainable
* Classifier trainable
* Layer4 learning rate: `1e-4`
* Classifier learning rate: `1e-3`

Both experiments use the same:

* Dataset
* Preprocessing
* Batch size
* Number of epochs
* Optimizer family
* Evaluation procedure
* Random seed

---

## Reproducibility

The experiment uses a fixed random seed:

```text
seed = 42
```

Configuration is centralized in `config.py`.

The experiment records:

* Random seed
* Batch size
* Number of epochs
* Learning rates
* Number of classes
* Trainable parameter count
* Best test accuracy
* Best epoch
* Training time

Results are saved to:

```text
artifacts/transfer_learning_results.json
```

---

## Experiment Results

Run:

```cmd
py compare_transfer_learning.py
```

Then:

```cmd
py report.py
```

The generated report compares:

| Experiment         | Trainable Parameters |        Best Accuracy |           Best Epoch |        Training Time |
| ------------------ | -------------------: | -------------------: | -------------------: | -------------------: |
| Feature Extraction | See generated report | See generated report | See generated report | See generated report |
| Fine-Tuning        | See generated report | See generated report | See generated report | See generated report |

The actual values are intentionally generated from the experiment rather than hard-coded.

---

## Engineering Analysis

Feature Extraction is significantly cheaper because only the final classifier is trained.

Fine-Tuning introduces substantially more trainable parameters, but it allows the pretrained high-level visual representation to adapt to the target dataset.

The preferred strategy should therefore be selected based on the measured accuracy improvement relative to the additional computational cost.

---

## Project Structure

```text
day24/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── checkpoint.py
├── experiments.py
├── reproducibility.py
├── main.py
├── compare_transfer_learning.py
├── report.py
├── experiment_report.md
│
├── tests/
│   ├── test_model.py
│   ├── test_optimizer.py
│   └── ...
│
└── artifacts/
    ├── checkpoints/
    └── transfer_learning_results.json
```

---

## Tests

Run:

```cmd
py -m pytest month01/day24/tests -v
```

All Day 24 tests should pass before committing the work.

---

## Key Engineering Lessons

### 1. Pretrained models are reusable components

A pretrained ResNet can provide a strong visual representation without training from scratch.

### 2. Freezing is a design decision

Not every pretrained layer needs to be updated.

### 3. Fine-Tuning should be controlled

Unfreezing too many parameters or using an excessive learning rate can damage pretrained representations.

### 4. Differential Learning Rate is useful

Newly initialized layers can use a larger learning rate, while pretrained layers use a smaller learning rate.

### 5. Experiments should be reproducible

A result is more useful when another person can reproduce the same experiment from the configuration and code.

### 6. Model selection should consider cost

The highest accuracy is not automatically the best engineering solution.

---

## Day 24 Outcome

By completing Day 24, the project has progressed from a basic CNN training pipeline toward a reusable computer-vision experiment system:

```text
CIFAR-10
   ↓
Pretrained ResNet18
   ↓
Feature Extraction
        vs
Partial Fine-Tuning
   ↓
Differential Learning Rate
   ↓
Evaluation
   ↓
Best Model
   ↓
Checkpoint
   ↓
Reproducible Experiment
   ↓
Experiment Report
```
