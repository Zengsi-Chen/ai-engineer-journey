# Medical Segmentation + XAI
## Architecture Documentation

## 1. System Overview

The project implements an end-to-end medical image segmentation
pipeline using Kvasir-SEG.

The current Day 34 system focuses on the training engine.

High-level pipeline:

Medical Image
    ↓
Dataset Validation
    ↓
Train / Validation / Test Split
    ↓
Preprocessing
    ↓
DataLoader
    ↓
U-Net
    ↓
BCE + Dice Loss
    ↓
AdamW
    ↓
Training / Validation
    ↓
Dice / IoU
    ↓
Best Model Checkpoint
    ↓
ReduceLROnPlateau
    ↓
Early Stopping
    ↓
Experiment Result
    ↓
Training Curves / Report


## 2. Project Layers

### Data Layer

Location:

`src/medseg/data/`

Responsibilities:

- dataset validation
- image/mask pairing
- train/validation/test splitting
- preprocessing
- tensor conversion
- DataLoader construction

Important design decision:

Image masks are resized using nearest-neighbor interpolation.

Images use bilinear interpolation.

This prevents interpolation from creating invalid intermediate
mask labels.


## 3. Model Layer

Location:

`src/medseg/models/`

The current segmentation model is U-Net.

The architecture contains:

- encoder blocks
- max pooling
- bottleneck
- decoder blocks
- skip connections
- final 1x1 convolution

The model outputs raw logits.

Sigmoid is intentionally not applied inside the model.

This allows `BCEWithLogitsLoss` to perform the numerically stable
logit-to-probability operation during training.


## 4. Loss Layer

Location:

`src/medseg/training/losses.py`

The current loss is:

BCE + Dice

with:

- BCE weight = 0.5
- Dice weight = 0.5

BCE provides pixel-level classification supervision.

Dice loss directly encourages overlap between predicted masks
and target masks.


## 5. Training Layer

Location:

`src/medseg/training/`

Main components:

- `train_step.py`
- `epoch.py`
- `optimizer.py`
- `scheduler.py`
- `early_stopping.py`
- `checkpoint.py`
- `trainer.py`

The `Trainer` acts as the training controller.

Its responsibilities include:

1. running training epochs
2. running validation
3. collecting metrics
4. saving the best checkpoint
5. updating the learning-rate scheduler
6. performing early stopping
7. recording experiment history


## 6. Evaluation Layer

Location:

`src/medseg/evaluation/`

Current metrics:

- Dice
- IoU

The default segmentation threshold is 0.5.

The evaluation layer is intentionally separated from the loss layer.

Loss functions optimize the model.

Evaluation metrics measure model performance.


## 7. Checkpoint Strategy

The best model is selected according to validation Dice.

Checkpoint contents include:

- epoch
- model state
- optimizer state
- validation metrics

The optimizer state is stored so that future versions can support
resume-training functionality.


## 8. Learning Rate Scheduling

The current scheduler is:

ReduceLROnPlateau

Configuration:

- monitor: validation Dice
- mode: max
- factor: 0.5
- patience: 2
- minimum learning rate: 1e-6

The scheduler reduces the learning rate when validation Dice
stops improving.


## 9. Early Stopping

Early stopping monitors validation Dice.

Current configuration:

- patience: 3
- minimum delta: 0.001

The purpose is to prevent unnecessary CPU training after the
validation metric has stopped meaningfully improving.


## 10. Reproducibility

The experiment uses a fixed random seed.

The current seed is:

42

The seed is applied to:

- Python random
- NumPy
- PyTorch


## 11. CPU Optimization

The project is currently developed on a CPU-only environment.

Two U-Net configurations were benchmarked.

Baseline:

`(32, 64, 128, 256)`

CPU-oriented:

`(16, 32, 64, 128)`

The smaller model has approximately 75% fewer parameters and
approximately 2.46x higher measured throughput in the previous
CPU benchmark.

The smaller configuration is therefore used for CPU development
experiments, while the larger configuration remains the baseline
architecture.


## 12. Testing Strategy

Tests are organized around individual engineering components.

Current test coverage includes:

- project structure
- dataset validation
- mask validation
- split integrity
- dataset loading
- DataLoader integration
- model architecture
- model integration
- loss functions
- optimizer
- training step
- training epoch
- evaluation metrics
- checkpointing
- scheduler
- early stopping
- Trainer

The goal is to verify components independently before running
the complete training pipeline.


## 13. Artifact Strategy

Generated experiment artifacts are stored separately from source code.

Important artifacts include:

`artifacts/checkpoints/`

Best model checkpoints.

`artifacts/experiments/`

Structured experiment results.

`artifacts/figures/`

Training curves and benchmark visualizations.

`artifacts/reports/`

Human-readable experiment reports.


## 14. Current Architecture Status

Day 34 provides a complete training engine.

Implemented:

- dataset validation
- deterministic dataset splitting
- preprocessing
- DataLoader
- U-Net
- model factory
- BCE + Dice loss
- optimizer factory
- training step
- epoch training
- validation metrics
- checkpointing
- learning-rate scheduling
- early stopping
- Trainer
- experiment logging
- benchmark
- training curves
- automated report generation

Future layers:

Inference
    ↓
Error Analysis
    ↓
XAI
    ↓
Visualization
    ↓
FastAPI
    ↓
Docker