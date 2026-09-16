# Day 39 — Tversky Loss Experiment

## Objective

Test whether Tversky Loss can reduce under-segmentation by assigning a larger penalty to false negatives.

## Configuration

- Model: U-Net
- Features: (16, 32, 64, 128)
- Dataset: Kvasir-SEG
- Train/Validation/Test: 700/150/150
- Image size: 256
- Augmentation: none
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Seed: 42
- Loss: Tversky Loss
- Alpha: 0.3
- Beta: 0.7
- Smooth: 1.0
- Evaluation threshold: 0.55

## Results

| Metric | Baseline | Tversky | Change |
|---|---:|---:|---:|
| Mean Dice | 0.6213 | 0.5320 | -0.0893 |
| Mean IoU | 0.5055 | 0.4009 | -0.1046 |
| Mean Precision | 0.7230 | 0.5617 | -0.1613 |
| Mean Recall | 0.6615 | 0.6670 | +0.0055 |
| Prediction FG ratio | 12.8631% | 19.2861% | Increased |
| Target FG ratio | 16.5577% | 16.5577% | Unchanged |

## Conclusion

Tversky Loss with alpha=0.3 and beta=0.7 slightly improved recall but substantially reduced precision and overall segmentation quality.

The prediction foreground ratio increased from 12.86% to 19.29%, indicating that the model became more aggressive in predicting foreground regions.

The increase in recall was too small to compensate for the loss in precision.

The Tversky configuration is rejected as the new default.

The Day 37 BCE+Dice baseline remains the best configuration so far.

## Next Experiment

Investigate false-positive control, threshold robustness, boundary quality, and targeted post-processing.

## Second Experiment: Tversky alpha=0.4, beta=0.6

The second Tversky configuration reduced the false-negative penalty:

- Alpha: 0.4
- Beta: 0.6
- Smooth: 1.0
- Locked evaluation threshold: 0.55

### Test Results

| Metric | Baseline | Tversky 0.3/0.7 | Tversky 0.4/0.6 |
|---|---:|---:|---:|
| Mean Dice | 0.6213 | 0.5320 | 0.5086 |
| Mean IoU | 0.5055 | 0.4009 | 0.3699 |
| Mean Precision | 0.7230 | 0.5617 | 0.4544 |
| Mean Recall | 0.6615 | 0.6670 | 0.7861 |
| Prediction FG ratio | 12.8631% | 19.2861% | 27.8640% |
| Target FG ratio | 16.5577% | 16.5577% | 16.5577% |

### Final Interpretation

The second configuration increased recall but caused a larger increase in false-positive predictions.

The prediction foreground ratio increased to 27.8640%, substantially exceeding the target foreground ratio of 16.5577%.

Mean Dice decreased further to 0.5086.

Both Tversky configurations are rejected. The original BCE+Dice baseline remains the best configuration.

Future experiments should investigate false-positive control, threshold robustness, boundary quality, and targeted post-processing rather than continuing to increase the false-negative penalty.
