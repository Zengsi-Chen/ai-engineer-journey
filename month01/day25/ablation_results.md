# ResNet-18 Ablation Study

## Objective

Investigate the effect of Skip Connections and Batch Normalization
on ResNet-18 training.

## Experimental Setup

- Dataset: CIFAR-10
- Training samples: 2000
- Validation samples: 500
- Batch size: 128
- Epochs: 5
- Optimizer: Adam
- Learning rate: 0.001
- CPU threads: 4
- DataLoader workers: 0
- Random seed: 42

## Results

| Model | Skip | BatchNorm | Best Val Accuracy | Final Train Accuracy | Final Val Accuracy |
|---|---:|---:|---:|---:|---:|
| Standard ResNet-18 | ✓ | ✓ | 0.3040 | 0.3765 | 0.3040 |
| No-Skip ResNet-18 | ✗ | ✓ | 0.2160 | 0.2465 | 0.2160 |
| No-BN ResNet-18 | ✓ | ✗ | 0.2220 | 0.2105 | 0.2220 |

## Analysis

### Skip Connection

The Standard ResNet-18 was compared with the No-Skip version while
keeping the other major components unchanged.

The experiment suggests that the skip connection improves optimization
and information flow in the deep network.

### Batch Normalization

The Standard ResNet-18 was compared with the No-BN version.

The experiment suggests that Batch Normalization improves training
stability and optimization under the current experimental settings.

## Limitations

This experiment used a small subset of CIFAR-10, only 5 epochs, and a
single random seed. Therefore, the results should be interpreted as an
engineering demonstration rather than a statistically rigorous study.

## Conclusion

The ablation study demonstrates that both Skip Connections and Batch
Normalization are important components of the ResNet architecture.