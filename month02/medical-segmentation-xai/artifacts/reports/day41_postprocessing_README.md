# Day 41 — Connected-Component Post-processing

## Objective

Evaluate whether inference-time connected-component filtering can reduce small false-positive regions and improve segmentation quality without retraining the model.

The experiment focuses on post-processing rather than changing the learned model.

---

## Experimental Design

### Pipeline

Image
→ U-Net
→ Probability Map
→ Threshold 0.55
→ Raw Binary Mask
→ Connected-Component Filtering
→ Final Segmentation

### Dataset

- Dataset: Kvasir-SEG
- Train: 700 images
- Validation: 150 images
- Test: 150 images
- Image size: 256 × 256
- Augmentation: None

### Model

- Architecture: U-Net
- Features: (16, 32, 64, 128)
- Checkpoint: `artifacts/checkpoints/best_model.pt`

### Locked Inference Configuration

- Threshold: 0.55
- Threshold source: Day 40 validation analysis
- Connectivity: 8-connected components

### Validation Candidates

The following minimum component areas were evaluated on the validation set:

- 0 pixels — baseline
- 50 pixels
- 100 pixels
- 200 pixels
- 500 pixels

The minimum component area was selected using validation Dice only.

---

## Validation Results

| Min Component Area | Dice | IoU | Precision | Recall | Prediction FG Ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6366 | 0.4995 | 0.6437 | 0.7485 | 0.1942 |
| 50 | 0.6366 | 0.4996 | 0.6447 | 0.7482 | 0.1940 |
| 100 | 0.6368 | 0.4998 | 0.6463 | 0.7481 | 0.1939 |
| 200 | **0.6370** | 0.5002 | **0.6470** | 0.7480 | 0.1936 |
| 500 | 0.6369 | **0.5007** | 0.6432 | 0.7469 | 0.1929 |

### Selected Configuration

- Threshold: 0.55
- Minimum component area: 200 pixels
- Validation Dice: 0.6370

The selected configuration was then locked before test evaluation.

---

## Locked Test Evaluation

No parameter optimization was performed on the test set.

### Test Configuration

- Threshold: 0.55
- Minimum component area: 200 pixels
- Test samples: 150

### Test Results

| Metric | Baseline | Post-processing |
|---|---:|---:|
| Mean Dice | 0.6144 | **0.6151** |
| Mean IoU | 0.4812 | **0.4822** |
| Mean Precision | 0.5982 | **0.5997** |
| Mean Recall | 0.7654 | 0.7642 |

### Test Metric Changes

- Dice: +0.0007
- IoU: +0.0010
- Precision: +0.0015
- Recall: -0.0012

Prediction foreground ratio after post-processing:

- Prediction: 18.5894%
- Target: 16.5577%

---

## Interpretation

Connected-component filtering produced a small improvement in test Dice, IoU, and precision, while recall decreased slightly.

The magnitude of the improvement is very small.

This suggests that small isolated false-positive components exist, but they are not the dominant source of segmentation error in this baseline model.

Therefore, connected-component filtering should be considered a lightweight inference-time refinement rather than a major performance improvement.

---

## Relation to Previous Experiments

### Day 39 — Tversky Loss

Increasing the false-negative penalty increased recall but substantially reduced precision and overall segmentation quality.

This indicated that simply encouraging more foreground predictions was not an effective solution.

### Day 40 — Threshold Robustness

Validation Dice remained very stable across thresholds from approximately 0.45 to 0.65.

This suggested that threshold selection was not the dominant source of error.

### Day 41 — Post-processing

Connected-component filtering produced only a marginal improvement.

Together, these experiments suggest that the main performance bottleneck is likely within the model's learned segmentation behavior rather than the final threshold or removal of small isolated regions.

---

## Engineering Lessons

1. Model output does not necessarily equal the final production prediction.
2. Post-processing can provide controlled inference-time improvements without retraining.
3. Post-processing parameters must be selected on validation data.
4. Test data must remain locked until the configuration is finalized.
5. Small numerical improvements should not automatically be interpreted as meaningful model improvements.
6. Controlled negative or low-impact experiments are useful for identifying where the real bottleneck is.

---

## Reproducibility

### Scripts

- `scripts/analyze_postprocessing.py`
- `scripts/evaluate_postprocessing_test.py`

### Implementation

- `src/medseg/postprocessing/connected_components.py`

### Tests

- `tests/test_postprocessing.py`

### Generated Reports

- `artifacts/reports/day41_postprocessing_validation.csv`
- `artifacts/reports/day41_postprocessing_validation.json`
- `artifacts/reports/day41_postprocessing_per_image.csv`
- `artifacts/reports/day41_postprocessing_test.csv`
- `artifacts/reports/day41_postprocessing_test.json`

---

## Final Conclusion

The locked experiment shows that removing connected components smaller than 200 pixels provides only a marginal improvement on the held-out test set.

The experiment therefore does not justify treating connected-component filtering as a major model improvement.

Its primary value is diagnostic: it provides evidence that isolated small regions are not the main limitation of the current segmentation system.

Future improvements should therefore focus on model learning, difficult-case segmentation, localization, and boundary quality rather than continued tuning of connected-component size alone.