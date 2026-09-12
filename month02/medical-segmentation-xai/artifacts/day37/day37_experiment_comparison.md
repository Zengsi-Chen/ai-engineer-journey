# Day 37 — Data Augmentation Experiment Comparison

## 1. Experiment Objective

The goal of Day 37 was to investigate whether geometric data augmentation improves
medical image segmentation performance and reduces localization failures.

Two controlled experiments were compared:

- Baseline: no augmentation
- Aug-A: horizontal flip augmentation

All other training settings were kept unchanged.

---

## 2. Experimental Setup

| Parameter | Baseline | Aug-A HFlip |
|---|---|---|
| Model | U-Net | U-Net |
| Features | (16, 32, 64, 128) | (16, 32, 64, 128) |
| Image Size | 256 × 256 | 256 × 256 |
| Batch Size | 4 | 4 |
| Epochs | 10 | 10 |
| Optimizer | AdamW | AdamW |
| Learning Rate | 1e-3 | 1e-3 |
| Weight Decay | 1e-4 | 1e-4 |
| Loss | BCE + Dice | BCE + Dice |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Threshold | 0.55 | 0.55 |
| Augmentation | None | Horizontal Flip |

---

## 3. Validation Results

| Metric | Baseline | Aug-A HFlip |
|---|---:|---:|
| Best Val Dice | 0.6329 | 0.5937 |
| Best Epoch | 10 | 9 |
| Early Stopping | No | No |

The horizontal-flip experiment produced a lower best validation Dice score
than the baseline.

---

## 4. Test Set Results

| Metric | Baseline | Aug-A HFlip | Change |
|---|---:|---:|---:|
| Dice | 0.6213 | 0.5766 | -0.0447 |
| IoU | 0.5055 | 0.4472 | -0.0583 |
| Precision | 0.7230 | 0.5981 | -0.1249 |
| Recall | 0.6615 | 0.6928 | +0.0313 |
| Prediction FG Ratio | 12.8633% | 17.0924% | +4.2291 pp |
| Target FG Ratio | 16.5577% | 16.5577% | 0 |

---

## 5. Analysis

### 5.1 Overall Segmentation Quality

Horizontal flipping did not improve segmentation quality.

Test Dice decreased from 0.6213 to 0.5766, while IoU decreased
from 0.5055 to 0.4472.

Therefore, under the current dataset and training configuration,
HFlip-only augmentation is not beneficial.

### 5.2 Precision-Recall Trade-off

The most important observation is the change in precision and recall.

Precision decreased substantially:

- Baseline: 0.7230
- Aug-A: 0.5981

Recall increased:

- Baseline: 0.6615
- Aug-A: 0.6928

This indicates that the augmented model became more aggressive in predicting
foreground pixels.

The prediction foreground ratio increased from 12.86% to 17.09%,
while the true foreground ratio remained 16.56%.

This is consistent with an increase in false-positive segmentation.

### 5.3 Relationship to Day 36 Error Analysis

Day 36 identified over-segmentation as the largest specific failure mode:

- Over-segmentation: 25.3%
- Under-segmentation: 9.3%
- Poor prediction: 8.7%
- Mixed/reasonable: 56.7%

The HFlip experiment did not reduce this problem.

Instead, its lower precision and larger predicted foreground area suggest
that HFlip may have made the over-segmentation tendency worse.

---

## 6. Engineering Conclusion

The experiment demonstrates that data augmentation should not be added
automatically just because it is considered a standard computer vision
practice.

For this segmentation task:

> HFlip-only augmentation decreased overall segmentation performance
> and increased the tendency toward foreground overprediction.

Therefore, the Baseline configuration remains the stronger model at this stage.

The result also demonstrates the value of controlled experimentation:
the augmentation hypothesis was tested rather than assumed to be correct.

---

## 7. Next Experiment

The next experiment should test a stronger augmentation policy:

### Aug-B

- Horizontal flip
- Vertical flip
- Random rotation ±15°

The purpose is to determine whether a richer geometric augmentation policy
can improve generalization without further increasing false-positive
segmentation.

The primary evaluation criteria should be:

1. Dice
2. IoU
3. Precision
4. Recall
5. Prediction foreground ratio
6. Over-segmentation rate
7. Dice = 0 rate

The key question is not simply:

> "Does augmentation increase Dice?"

but:

> "Does augmentation improve segmentation quality while reducing the
> localization and over-segmentation failures identified in Day 36?"

## 8. Final Comparison: Baseline vs Aug-A vs Aug-B

| Metric | Baseline | Aug-A HFlip | Aug-B Full |
|---|---:|---:|---:|
| Best Val Dice | 0.6329 | 0.5937 | 0.5803 |
| Test Dice | 0.6213 | 0.5766 | 0.5453 |
| Test IoU | 0.5055 | 0.4472 | 0.4138 |
| Test Precision | 0.7230 | 0.5981 | 0.5774 |
| Test Recall | 0.6615 | 0.6928 | 0.6424 |
| Prediction FG Ratio | 12.8633% | 17.0924% | 16.9967% |

Both augmentation experiments performed worse than the baseline.

The Baseline model achieved the strongest test Dice, IoU, and precision.
Aug-A increased recall but reduced precision, while Aug-B produced the
lowest overall segmentation quality.

The final Day 37 decision is to retain the Baseline configuration as the
current production candidate.

Future experiments should investigate the effect of augmentation strength,
rotation padding strategy, training budget, and loss design separately.