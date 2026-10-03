# Day 43 — Boundary-Aware Improvement Experiment

## 1. Objective

Evaluate whether adding boundary-aware supervision to the existing U-Net
training objective can improve segmentation boundary quality while
maintaining overall segmentation performance.

---

## 2. Research Question

Does adding a boundary-aware loss to the current BCE + Dice objective
improve boundary alignment without degrading overall segmentation quality?

---

## 3. Motivation

Day 42 boundary analysis showed that segmentation errors were strongly
associated with boundary disagreement.

Validation analysis found:

- Mean ASSD: 20.5254 pixels
- Mean Hausdorff: 85.8383 pixels
- Dice and ASSD correlation: -0.8144
- Dice and Hausdorff correlation: -0.4444

These findings motivated a controlled experiment targeting boundary quality.

---

## 4. Experimental Groups

### Group A — Baseline

Loss:

- BCE + Dice
- BCE weight: 0.5
- Dice weight: 0.5

### Group B — Boundary-aware

Loss:

- BCE + Dice
- Boundary Loss weight: 0.1
- Boundary pixel weight: 3.0

The Boundary Loss increases the contribution of pixels near the target
boundary.

---

## 5. Controlled Variables

The following variables were kept identical:

- Dataset: Kvasir-SEG
- Train/Validation/Test: 700/150/150
- Model: U-Net
- Features: (16, 32, 64, 128)
- Image size: 256 × 256
- Augmentation: none
- Batch size: 4
- Epochs: 10
- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Random seed: 42
- Device: CPU
- Segmentation threshold: 0.55

The only experimental variable was the addition of boundary-aware
supervision.

---

## 6. Validation Protocol

The best checkpoint for each model was selected using validation Dice.

### Baseline

- Best epoch: 10
- Best validation Dice: 0.636632
- Best validation IoU: 0.499555

### Boundary-aware

- Best epoch: 10
- Best validation Dice: 0.625528
- Best validation IoU: 0.487777

The validation threshold remained fixed at 0.55.

No test data was used for model or threshold selection.

---

## 7. Locked Test Protocol

After validation checkpoint selection, both models were evaluated once
on the same 150 test samples.

No retraining, threshold optimization, post-processing, or test-time
parameter tuning was performed.

---

## 8. Locked Test Results

| Metric | Baseline | Boundary-aware | Change |
|---|---:|---:|---:|
| Dice | 0.6144 | 0.6100 | -0.0044 |
| IoU | 0.4812 | 0.4764 | -0.0049 |
| Precision | 0.5982 | 0.6020 | +0.0038 |
| Recall | 0.7654 | 0.7531 | -0.0123 |
| ASSD | 23.1178 | 21.3035 | -1.8143 |
| Median ASSD | 20.4360 | 17.6487 | -2.7873 |
| Hausdorff | 82.2666 | 79.6516 | -2.6151 |
| Median Hausdorff | 83.6720 | 76.1662 | -7.5052 |

Lower values are better for ASSD and Hausdorff.

---

## 9. Paired Test Audit

The two models were compared sample-by-sample using the same test IDs.

Total paired samples:

- 150

### Dice

- Improved: 77
- Worsened: 73
- Unchanged: 0

### IoU

- Improved: 77
- Worsened: 73
- Unchanged: 0

### ASSD

Valid paired samples:

- 147
- Improved: 86
- Worsened: 61
- Unchanged: 0

### Hausdorff

Valid paired samples:

- 147
- Improved: 91
- Worsened: 54
- Unchanged: 2

The paired analysis confirms that the reduction in boundary distance
was not caused by a single extreme sample.

---

## 10. Interpretation

The experiment produced a clear trade-off.

Boundary-aware supervision improved boundary-distance metrics:

- Mean ASSD decreased from 23.1178 to 21.3035.
- Mean Hausdorff decreased from 82.2666 to 79.6516.
- ASSD improved in 86 of 147 valid paired cases.
- Hausdorff improved in 91 of 147 valid paired cases.

However, overall overlap performance decreased slightly:

- Dice decreased by 0.0044.
- IoU decreased by 0.0049.
- Recall decreased by 0.0123.

Precision increased slightly by 0.0038.

This indicates that the boundary-aware model became somewhat more
conservative in foreground prediction. Boundary alignment improved,
but some foreground regions were no longer recovered.

---

## 11. Comparison With Previous Experiments

### Day 39 — Tversky Loss

Tversky loss increased emphasis on false negatives.

Observed behavior:

- Recall increased slightly with alpha=0.3, beta=0.7.
- Precision and Dice decreased.
- Stronger FN weighting caused substantial foreground expansion.
- Overall segmentation quality deteriorated.

### Day 41 — Connected Components

Post-processing with a minimum component area of 200 pixels produced
only marginal test improvement.

This suggested that isolated small components were not the dominant
error source.

### Day 42 — Boundary Analysis

Boundary analysis showed that poor overlap was strongly associated
with large boundary disagreement.

This motivated Day 43.

### Day 43 — Boundary-aware Training

Boundary-aware supervision improved ASSD and Hausdorff but slightly
reduced Dice and IoU.

Therefore, the experiment provides evidence that boundary quality is
an actionable optimization target, but the current boundary loss
formulation does not improve the complete segmentation objective by
itself.

---

## 12. Conclusion

The boundary-aware experiment produced a meaningful trade-off rather
than a universal improvement.

The current boundary-aware loss:

- improved average boundary alignment;
- improved boundary metrics in a majority of valid paired cases;
- slightly improved precision;
- slightly reduced recall;
- slightly reduced Dice and IoU.

Therefore:

> Boundary-aware supervision improved spatial boundary alignment but
> did not improve overall segmentation overlap in the current U-Net
> configuration.

The experiment should be considered a useful controlled negative/trade-off
result rather than a failed experiment.

It demonstrates that optimizing a clinically meaningful error dimension
can improve that dimension while affecting other aspects of segmentation
quality.

---

## 13. Limitations

1. ASSD and Hausdorff are measured in resized 256 × 256 pixel
   coordinates, not physical millimeters.

2. Only one boundary-loss formulation and weighting were tested.

3. No augmentation was used.

4. The experiment used a single random seed.

5. The test set contains 150 samples.

6. Boundary metrics were unavailable for a small number of samples
   because one or both masks were empty.

7. No statistical significance testing was performed.

---

## 14. Reproducibility

Training configuration:

- Seed: 42
- CPU execution
- U-Net features: (16, 32, 64, 128)
- Image size: 256
- Batch size: 4
- Epochs: 10
- AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Threshold: 0.55

Validation-selected checkpoints:

- `artifacts/day43/baseline/best_model.pt`
- `artifacts/day43/boundary_01/best_model.pt`

Test outputs:

- `artifacts/reports/day43_baseline_test_per_image.csv`
- `artifacts/reports/day43_baseline_test_summary.json`
- `artifacts/reports/day43_boundary_01_test_per_image.csv`
- `artifacts/reports/day43_boundary_01_test_summary.json`
- `artifacts/reports/day43_paired_test_audit.csv`

---

## 15. Engineering Takeaway

The main engineering lesson from Day 43 is that model improvement should
not be reduced to optimizing a single metric.

A segmentation system can simultaneously:

- improve boundary alignment,
- improve precision,
- reduce recall,
- and reduce Dice.

Therefore, model evaluation should consider multiple complementary
metrics and should use controlled validation, locked test evaluation,
and paired sample-level analysis.

Day 43 establishes boundary-aware supervision as a measurable but
currently trade-off-heavy improvement direction for this project.