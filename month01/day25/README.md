# Day 25 — ResNet Architecture and CNN Deep Dive

## Overview

Day 25 focused on understanding and implementing deep convolutional neural networks through ResNet-18.

The project moved beyond simply using a prebuilt model. It investigated the engineering and optimization effects of two important architectural components:

* Skip Connections
* Batch Normalization

The final project includes:

* ResNet-18 implementation
* CIFAR-10 training
* CPU performance benchmarking
* CPU thread optimization
* Data augmentation experiments
* Model ablation studies
* Structured experiment result tracking
* Training curve visualization
* Evidence-based experiment analysis

---

## Learning Objectives

The main goal of this project was to understand:

```text
Deep CNN
    ↓
Degradation Problem
    ↓
Residual Learning
    ↓
Skip Connection
    ↓
ResNet Architecture
    ↓
Batch Normalization
    ↓
Optimization
    ↓
Ablation Study
    ↓
Experiment Tracking
```

---

## ResNet Residual Learning

A traditional neural network attempts to learn:

```text
H(x)
```

A residual block instead learns:

```text
F(x) = H(x) - x
```

The output becomes:

```text
H(x) = F(x) + x
```

In code:

```python
out = block(x)
out = out + shortcut(x)
```

This is called a Skip Connection or Residual Connection.

Skip Connections improve information and gradient flow and help make deep neural networks easier to optimize.

---

## ResNet-18 Architecture

The project implements a ResNet-style architecture for CIFAR-10.

```text
Input
  ↓
Conv + BatchNorm + ReLU
  ↓
Residual Layer 1
  ↓
Residual Layer 2
  ↓
Residual Layer 3
  ↓
Residual Layer 4
  ↓
Global Average Pooling
  ↓
Fully Connected Layer
  ↓
10 Classes
```

Each residual block contains:

```text
Input
  │
  ├──────────── Shortcut ────────┐
  │                             │
  ↓                             │
Conv → BatchNorm → ReLU          │
  ↓                             │
Conv → BatchNorm                 │
  ↓                             │
Add ◄────────────────────────────┘
  ↓
ReLU
```

---

## Dataset

The project uses CIFAR-10.

```text
10 classes
32 × 32 RGB images
```

The experiment uses a reduced dataset size to make training practical on CPU.

### CPU Experiment Configuration

```text
Training samples: 2000
Validation samples: 500
Batch size: 128
Epochs: 5
Optimizer: Adam
Learning rate: 0.001
CPU threads: 4
DataLoader workers: 0
Random seed: 42
```

---

## CPU Performance Benchmark

The project benchmarks different CPU thread settings.

Example result:

```text
Threads: 1
Time: 190.67 seconds
Throughput: 5.24 samples/sec

Threads: 2
Time: 117.53 seconds
Throughput: 8.51 samples/sec

Threads: 4
Time: 101.97 seconds
Throughput: 9.81 samples/sec
```

The experiment showed that 4 CPU threads provided the highest throughput on the test machine.

Therefore:

```text
num_threads = 4
```

was selected for the CPU experiments.

---

## Data Augmentation

The project explores data augmentation for CIFAR-10.

Typical augmentation operations include:

```text
Random Crop
Random Horizontal Flip
Normalization
```

Data augmentation increases the diversity of training samples and can improve generalization.

---

## Ablation Study

Three models were compared.

| Model              | Skip Connection | BatchNorm |
| ------------------ | --------------: | --------: |
| Standard ResNet-18 |               ✓ |         ✓ |
| No-Skip ResNet-18  |               ✗ |         ✓ |
| No-BN ResNet-18    |               ✓ |         ✗ |

The purpose of the ablation study was to isolate the impact of major ResNet components.

### Skip Connection Experiment

```text
Standard ResNet-18
        vs
No-Skip ResNet-18
```

The main architectural difference is:

```text
Skip Connection
✓ → ✗
```

This experiment investigates whether residual connections improve deep network optimization.

### Batch Normalization Experiment

```text
Standard ResNet-18
        vs
No-BN ResNet-18
```

The main architectural difference is:

```text
BatchNorm
✓ → ✗
```

This experiment investigates the impact of normalization on training stability and optimization.

---

## Experiment Result Tracking

Each training epoch is stored using a structured result object.

```text
ExperimentResult
│
├── name
├── epoch_results
│
│   ├── EpochResult
│   │   ├── train_loss
│   │   ├── train_accuracy
│   │   ├── validation_loss
│   │   └── validation_accuracy
│
├── best_validation_accuracy
└── training_time
```

This makes experiment analysis reproducible.

Results are saved as:

```text
artifacts/ablation_results.json
```

---

## Training Curves

The project generates:

```text
Validation Loss
Training Accuracy
Validation Accuracy
```

Example artifact structure:

```text
artifacts/
├── ablation_results.json
├── ablation_validation_loss.png
├── ablation_training_accuracy.png
└── ablation_validation_accuracy.png
```

Training curves allow analysis of:

```text
Optimization Speed
Training Stability
Validation Performance
Overfitting
Underfitting
Generalization Gap
```

---

## Generalization Gap

The final generalization gap is calculated as:

```text
Train Accuracy
-
Validation Accuracy
```

A small gap alone does not necessarily indicate a better model.

For example:

```text
Model A
Train Accuracy: 80%
Validation Accuracy: 60%

Model B
Train Accuracy: 35%
Validation Accuracy: 32%
```

Model B has a smaller gap, but it may simply be underfitting.

Therefore, model performance should be evaluated using:

```text
Training Accuracy
+
Validation Accuracy
+
Validation Loss
+
Training Curves
```

rather than one metric alone.

---

## Key Engineering Lessons

### 1. Best accuracy is not enough

```text
Best Accuracy
        ↓
Final Result

Training Curves
        ↓
Training Process
```

Training curves reveal how a model learns, not just its final score.

### 2. Skip Connections improve optimization

Residual connections provide an identity path:

```text
Input
  ↓
F(x) + x
  ↓
Output
```

This improves information and gradient flow in deep networks.

### 3. Batch Normalization improves training

Batch Normalization can improve:

```text
Training Stability
Optimization
Learning Speed
```

### 4. Ablation studies are essential

Instead of asking:

```text
"Which model is better?"
```

Ablation asks:

```text
"What caused the performance difference?"
```

This is a more scientific approach to model engineering.

### 5. Experiments should produce artifacts

The final workflow is:

```text
Model
  ↓
Training
  ↓
Structured Metrics
  ↓
ExperimentResult
  ↓
JSON Artifact
  ↓
Visualization
  ↓
Analysis
```

This represents a small experiment tracking system.

---

## Run Instructions

### Run tests

```cmd
py -m pytest -q
```

### Check Python syntax

```cmd
py -m compileall -q .
```

### Run CPU benchmark

```cmd
py benchmark_cpu.py
```

### Run ablation experiments

```cmd
py run_all_ablations.py
```

### Plot validation loss

```cmd
py plot_ablation_loss.py
```

### Plot accuracy curves

```cmd
py plot_ablation_accuracy.py
```

### Analyze ablation results

```cmd
py analyze_ablation.py
```

---

## Final Learning Outcome

Day 25 progressed from basic CNN knowledge to deep CNN architecture and experimental analysis.

```text
CNN
 ↓
Deep CNN
 ↓
Degradation Problem
 ↓
Residual Learning
 ↓
Skip Connection
 ↓
ResNet-18
 ↓
CIFAR-10
 ↓
Data Augmentation
 ↓
BatchNorm
 ↓
CPU Optimization
 ↓
Ablation Study
 ↓
Experiment Tracking
 ↓
Training Curves
 ↓
Evidence-Based Analysis
```

The key outcome was learning not only how to train a ResNet, but how to investigate why an architectural component affects training behavior.

## Experimental Results

========================================================
RESNET ABLATION ANALYSIS
========================================================

Model: Standard ResNet-18
Best Val Accuracy: 0.3040
Final Train Accuracy: 0.3765
Final Val Accuracy: 0.3040
Final Val Loss: 1.9411
Generalization Gap: 0.0725
Training Time: 634.95s

Model: No-Skip ResNet-18
Best Val Accuracy: 0.2160
Final Train Accuracy: 0.2465
Final Val Accuracy: 0.2160
Final Val Loss: 2.3547
Generalization Gap: 0.0305
Training Time: 599.29s

Model: No-BN ResNet-18
Best Val Accuracy: 0.2220
Final Train Accuracy: 0.2105
Final Val Accuracy: 0.2220
Final Val Loss: 2.0401
Generalization Gap: -0.0115
Training Time: 615.07s

========================================================
BEST MODEL
========================================================
Standard ResNet-18
Best Validation Accuracy: 0.3040
