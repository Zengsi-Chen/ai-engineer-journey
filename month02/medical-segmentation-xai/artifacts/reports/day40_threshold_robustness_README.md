# Day 40 — Threshold Robustness & Operating Point Analysis

## Objective

Evaluate the robustness of the segmentation operating point with respect to
the prediction threshold.

The analysis does not retrain the model and does not use the test set for
threshold selection.

The threshold was previously optimized on the validation set during Day 35.
Day 40 analyzes the stability of that selected operating point.

---

## Experimental Setup

- Dataset: Kvasir-SEG
- Train / Validation / Test: 700 / 150 / 150
- Model: U-Net
- Features: (16, 32, 64, 128)
- Image size: 256 × 256
- Augmentation: None
- Threshold candidates: 0.10–0.90
- Threshold step: 0.05
- Threshold selection set: Validation set
- Test set: Not used for threshold optimization

---

## Source Experiment

Day 35 performed global threshold optimization on the validation set.

The validation results are stored in:

`artifacts/reports/day35_threshold_optimization.csv`

Day 40 reuses these results rather than retraining or re-evaluating the model.

---

## Validation-Optimal Operating Point

The maximum validation Dice was obtained at:

- Threshold: **0.55**
- Dice: **0.636610**
- IoU: **0.499514**
- Precision: **0.643713**
- Recall: **0.748518**
- Foreground ratio: **0.194243**

Therefore:

**Selected operating threshold = 0.55**

---

## Threshold Robustness

### 0.50–0.60

Dice range:

**0.000529**

Maximum Dice degradation relative to the optimum:

**0.000529**

### 0.45–0.65

Dice range:

**0.002196**

Maximum Dice degradation relative to the optimum:

**0.002196**

The very small Dice variation indicates that segmentation quality is highly
stable around the selected operating point.

---

## Operating Point Trade-off

| Threshold | Dice | Precision | Recall | Foreground Ratio |
|-----------|------|-----------|--------|------------------|
| 0.45 | 0.635578 | 0.623928 | 0.771963 | 0.207878 |
| 0.50 | 0.636543 | 0.634066 | 0.760618 | 0.200995 |
| 0.55 | 0.636610 | 0.643713 | 0.748518 | 0.194243 |
| 0.60 | 0.636081 | 0.653540 | 0.735728 | 0.187438 |
| 0.65 | 0.634414 | 0.663972 | 0.720991 | 0.180237 |

Increasing the threshold produces:

- Higher precision
- Lower recall
- Lower predicted foreground area

This demonstrates that the threshold acts primarily as an operating-point
control between false-positive and false-negative behavior.

---

## Relationship to Day 39

Day 39 evaluated Tversky Loss with increased false-negative penalty.

The Tversky experiments increased recall but substantially reduced precision
and overall Dice.

Day 40 demonstrates an alternative way to control the precision-recall
operating point without retraining the model.

This provides a useful engineering distinction:

- **Loss function** changes model training behavior.
- **Prediction threshold** changes inference-time operating behavior.

For the current model, threshold adjustment provides a more controlled way to
modify the precision-recall trade-off than increasing the false-negative
penalty during training.

---

## Data Integrity Check

Day 40 results were compared directly with the Day 35 source CSV.

Validation rows: **17**

Source/analysis threshold and Dice values:

**Identical**

This confirms that Day 40 performs a post-hoc robustness analysis without
altering the original Day 35 experiment results.

---

## Conclusion

The validation-optimal threshold is **0.55**.

More importantly, the operating point is not sensitive to small threshold
changes.

The Dice score remains nearly unchanged across 0.50–0.60 and changes by only
0.002196 across 0.45–0.65.

Therefore, the model has a **robust threshold operating region rather than a
sharp single-point optimum**.

The threshold of 0.55 is retained as the locked inference threshold for
subsequent test evaluation.

---

## Engineering Takeaways

1. Threshold selection must be performed on the validation set.
2. The test set should remain locked until the operating point is finalized.
3. A flat threshold-response curve is preferable to a sharp isolated optimum
   because small threshold perturbations have limited impact.
4. Precision-recall trade-offs can be adjusted at inference time without
   retraining.
5. The current model does not show evidence that further increasing the
   false-negative penalty is beneficial.
6. Future improvements should investigate false-positive control, boundary
   quality, targeted post-processing, and model architecture rather than
   blindly increasing the FN penalty.

---

## Artifacts

- `artifacts/reports/day40_threshold_robustness.csv`
- `artifacts/reports/day40_threshold_robustness.json`
- `artifacts/figures/day40_threshold_robustness.png`
- `artifacts/reports/day40_threshold_robustness_README.md`

---

## Reproducibility

Run:

```cmd
py scripts\analyze_threshold_robustness.py