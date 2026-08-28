# Transfer Learning Experiment

## Objective

Compare Feature Extraction and Fine-Tuning
using a pretrained ResNet18 on CIFAR-10.

## Experimental Setup

- Dataset: CIFAR-10
- Model: ResNet18 pretrained on ImageNet
- Batch Size: 64
- Epochs: 5
- Seed: 42
- Optimizer: Adam

## Experiments

### Feature Extraction

- Backbone: Frozen
- Classifier: Trainable
- Learning Rate: 1e-3

### Fine-Tuning

- Layer1: Frozen
- Layer2: Frozen
- Layer3: Frozen
- Layer4: Trainable
- Layer4 LR: 1e-4
- Classifier LR: 1e-3

## Results

| Experiment | Trainable Parameters | Best Accuracy | Best Epoch | Training Time |
|---|---:|---:|---:|---:|
| Feature Extraction | ... | ... | ... | ... |
| Fine-Tuning | ... | ... | ... | ... |

## Analysis

Fine-Tuning changed the pretrained high-level
representation by unfreezing Layer4.

Feature Extraction trained only the final classifier,
resulting in substantially fewer trainable parameters.

## Conclusion

Based on the measured accuracy and computational cost,
the preferred approach for this experiment is: ...

## Reproducibility

Results were generated using:

- Seed: 42
- Batch Size: 64
- Epochs: 5
- Classifier LR: 1e-3
- Backbone LR: 1e-4