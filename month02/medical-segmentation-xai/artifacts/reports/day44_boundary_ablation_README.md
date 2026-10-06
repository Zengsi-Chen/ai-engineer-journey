# Day 44 — Boundary Loss Weight Ablation

## 1. Research Question

Day 43 showed that adding Boundary Loss improved boundary quality but slightly reduced overall segmentation overlap.

Day 44 investigates whether this trade-off is caused by the Boundary Loss itself or by the selected loss weight.

The experiment tests four Boundary Loss weights:

* 0.00 — Baseline BCE + Dice
* 0.05 — Boundary-aware
* 0.10 — Boundary-aware
* 0.20 — Boundary-aware

The objective is to identify whether a moderate Boundary Loss weight can improve boundary quality while preserving overall segmentation performance.

---

## 2. Experimental Design

### Dataset

* Dataset: Kvasir-SEG
* Train: 700
* Validation: 150
* Test: 150
* Image size: 256 × 256
* Train/validation/test splits are disjoint.

### Model

* Architecture: U-Net
* Features: (16, 32, 64, 128)
* Input channels: 3
* Output channels: 1

### Training

* Optimizer: AdamW
* Learning rate: 1e-3
* Weight decay: 1e-4
* Batch size: 4
* Epochs: 10
* Seed: 42
* Device: CPU
* Data augmentation: none

### Loss

Baseline:

`BCE + Dice`

Boundary-aware:

`BCE + Dice + boundary_loss_weight × BoundaryLoss`

BoundaryLoss uses boundary pixels from the target mask and assigns higher loss weight to those pixels.

The internal boundary-pixel weight is fixed at 3.0.

### Evaluation

The segmentation threshold was locked at:

`threshold = 0.55`

The threshold was selected previously during Day 40 validation analysis and was not re-optimized during Day 44.

---

## 3. Validation Ablation

The four configurations were trained independently while keeping all other variables fixed.

| Configuration | Boundary Loss Weight | Val Dice | Val IoU | Val Precision | Val Recall | Val ASSD | Val Hausdorff |
| ------------- | -------------------: | -------: | ------: | ------------: | ---------: | -------: | ------------: |
| Baseline      |                 0.00 |   0.6366 |  0.4996 |        0.6437 |     0.7485 |  20.5254 |       85.8383 |
| Boundary 0.05 |                 0.05 |   0.6086 |  0.4749 |        0.6752 |     0.6596 |  20.3815 |       79.6492 |
| Boundary 0.10 |                 0.10 |   0.6255 |  0.4878 |        0.6538 |     0.7255 |  19.7916 |       77.8190 |
| Boundary 0.20 |                 0.20 |   0.6270 |  0.4879 |        0.6576 |     0.7047 |  20.0645 |       79.0659 |

### Validation Interpretation

The effect of Boundary Loss weight is non-monotonic.

The 0.05 configuration produced the largest degradation in Dice and IoU.

Increasing the weight to 0.10 recovered part of the overlap performance while producing the best boundary metrics.

Increasing the weight further to 0.20 slightly improved Dice relative to 0.10, but boundary quality became worse.

Therefore, 0.10 provides the best boundary-quality operating point among the tested configurations.

The weight was selected using validation results only.

---

## 4. Locked Test Evaluation

After validation selection, the Boundary Loss weight was locked at:

`0.10`

No test-set tuning or parameter optimization was performed.

The baseline and boundary-aware models were evaluated on the same 150 test images using the fixed threshold of 0.55.

### Test Results

| Metric      | Baseline | Boundary 0.10 |  Change |
| ----------- | -------: | ------------: | ------: |
| Dice ↑      |   0.6144 |        0.6100 | -0.0044 |
| IoU ↑       |   0.4812 |        0.4764 | -0.0049 |
| Precision ↑ |   0.5982 |        0.6020 | +0.0038 |
| Recall ↑    |   0.7654 |        0.7531 | -0.0123 |
| ASSD ↓      |  23.1178 |       21.3035 | -1.8143 |
| Hausdorff ↓ |  82.2666 |       79.6516 | -2.6151 |

Boundary metrics were valid for 149 baseline samples and 148 boundary-aware samples.

---

## 5. Paired Test Audit

The two models were evaluated on exactly the same 150 test samples.

### Paired Improvement Counts

| Metric    | Improved | Worsened | Unchanged | Valid |
| --------- | -------: | -------: | --------: | ----: |
| Dice      |       77 |       73 |         0 |   150 |
| IoU       |       77 |       73 |         0 |   150 |
| ASSD      |       86 |       61 |         0 |   147 |
| Hausdorff |       91 |       54 |         2 |   147 |

For ASSD:

* 86 / 147 samples improved
* 61 / 147 samples worsened

For Hausdorff:

* 91 / 147 samples improved
* 54 / 147 samples worsened
* 2 / 147 samples were unchanged

Thus, the improvement in boundary quality is not explained solely by a small number of extreme cases.

---

## 6. Validation-to-Test Consistency

The validation and test sets show the same overall direction.

### Validation

Boundary-aware 0.10 compared with baseline:

* Dice decreased
* IoU decreased
* ASSD decreased
* Hausdorff decreased

### Test

Boundary-aware 0.10 compared with baseline:

* Dice decreased by 0.0044
* IoU decreased by 0.0049
* ASSD decreased by 1.8143
* Hausdorff decreased by 2.6151

The consistency of the direction between validation and locked test evaluation provides evidence that the observed trade-off generalizes beyond the validation set.

---

## 7. Final Conclusion

Boundary Loss improves spatial boundary alignment but does not improve overall segmentation overlap in the current U-Net configuration.

The selected weight of 0.10 produced the best boundary quality among the tested weights on validation and retained this boundary-quality improvement on the locked test set.

However, the improvement comes with a small reduction in Dice and IoU.

Therefore:

> Boundary Loss is useful for improving spatial localization, but it is not sufficient as a standalone strategy for improving overall segmentation quality.

The experiment supports treating Boundary Loss as a complementary objective rather than a replacement for the baseline segmentation objective.

---

## 8. Why the Result Matters

The goal of this experiment was not simply to maximize Dice.

Day 44 separates two different aspects of segmentation quality:

1. **Region overlap**

   * Dice
   * IoU

2. **Spatial localization**

   * ASSD
   * Hausdorff distance

The experiment demonstrates that a model can become spatially better aligned with the target boundary while becoming slightly worse in region overlap.

This is important for medical image segmentation because boundary accuracy and region overlap are related but not identical properties.

---

## 9. Limitations

This experiment has several limitations:

* Only one dataset was used.
* Only one U-Net architecture was evaluated.
* No data augmentation was used.
* Only four Boundary Loss weights were tested.
* No statistical significance test was performed.
* The experiment uses a single fixed random seed.
* The test set was used only after validation-based weight selection.
* Boundary Loss itself is based on target-mask boundary weighting rather than a more explicit geometric or distance-transform formulation.

Therefore, the results should be interpreted as evidence for this experimental configuration rather than as a universal conclusion about Boundary Loss.

---

## 10. Engineering Takeaway

The Day 44 experiment demonstrates a complete controlled ML experiment workflow:

1. Identify a failure mode.
2. Introduce a targeted modeling change.
3. Define the primary and secondary metrics.
4. Perform validation-only ablation.
5. Select the operating point without using the test set.
6. Lock the configuration.
7. Evaluate once on the unseen test set.
8. Perform paired per-sample analysis.
9. Analyze both improvements and regressions.
10. Document the trade-off and limitations.

The final result is not simply "the new loss is better."

Instead:

> The boundary-aware objective produces a measurable spatial-quality improvement, but the improvement does not translate into better overall segmentation overlap.

This provides a concrete basis for the next research direction.
