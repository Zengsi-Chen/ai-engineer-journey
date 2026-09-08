# Medical Image Segmentation + XAI

An engineering-focused medical image segmentation project
built with PyTorch.

The project uses the Kvasir-SEG dataset and is designed as a
reproducible end-to-end AI engineering portfolio project.

## Current Status

### Completed

- Dataset validation
- Train / validation / test splitting
- Leakage-safe split verification
- Image and mask preprocessing
- DataLoader
- U-Net segmentation model
- Model factory
- BCE + Dice loss
- AdamW optimizer
- Training and validation engine
- Dice and IoU evaluation
- Best-model checkpointing
- ReduceLROnPlateau scheduler
- Early stopping
- Reusable Trainer
- Experiment result logging
- Training curves
- CPU model benchmarking
- Architecture documentation

### Planned

- Test-set evaluation
- Prediction pipeline
- Segmentation visualization
- Error analysis
- Grad-CAM / XAI
- Uncertainty estimation
- FastAPI inference service
- Docker deployment

## Architecture

```text
Medical Image
      ↓
Data Validation
      ↓
Dataset Split
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
Trainer
      ├── Validation
      ├── Dice / IoU
      ├── Checkpoint
      ├── Scheduler
      └── Early Stopping
      ↓
Experiment Artifacts
      ↓
Inference
      ↓
XAI
      ↓
FastAPI
      ↓
Docker