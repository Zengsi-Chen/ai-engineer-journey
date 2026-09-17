# Day 42 — Boundary-Aware Segmentation Analysis

## 1. Objective

The objective of Day 42 is to determine whether segmentation errors in the current U-Net model are associated with inaccurate target boundaries.

The analysis is performed entirely at inference time.

No model retraining, threshold optimization on the test set, or checkpoint modification is performed.

The analysis focuses on:

* Boundary extraction
* Average Symmetric Surface Distance (ASSD)
* Hausdorff distance
* Per-image boundary analysis
* Correlation between overlap and boundary metrics
* Boundary failure profiling
* Visual inspection of representative failure cases

---

## 2. Experimental Configuration

| Component                 | Configuration     |
| ------------------------- | ----------------- |
| Dataset                   | Kvasir-SEG        |
| Train / Validation / Test | 700 / 150 / 150   |
| Model                     | U-Net             |
| Features                  | (16, 32, 64, 128) |
| Image size                | 256 × 256         |
| Threshold                 | 0.55              |
| Augmentation              | None              |
| Device                    | CPU               |
| Validation samples        | 150               |

The threshold of 0.55 was selected previously using the validation set and is treated as locked for this analysis.

No test-set threshold optimization was performed.

---

## 3. Boundary Metric Implementation

Boundary analysis was implemented in:

```text
src/medseg/evaluation/boundary_metrics.py
```

The implementation uses:

* NumPy
* SciPy

No additional dependency such as scikit-image was introduced.

### Boundary extraction

A one-pixel inner boundary is extracted using binary erosion with an 8-connected 3 × 3 structure.

### ASSD

Average Symmetric Surface Distance measures the average nearest-boundary distance in both directions:

* prediction boundary → target boundary
* target boundary → prediction boundary

Lower values indicate better boundary agreement.

### Hausdorff distance

Hausdorff distance measures the maximum nearest-boundary distance between the two boundaries.

Lower values indicate better worst-case boundary agreement.

---

## 4. Important Measurement Limitation

All boundary distances reported in this experiment are measured in the resized 256 × 256 image coordinate system.

Therefore:

> ASSD and Hausdorff values are reported in pixels, not millimeters.

No physical pixel-spacing metadata was incorporated into this experiment.

Consequently, these measurements should be interpreted as image-space geometric errors.

---

## 5. Validation Results

The locked threshold of 0.55 was evaluated on the 150-image validation set.

| Metric    |       Mean |
| --------- | ---------: |
| Dice      |     0.6366 |
| IoU       |     0.4996 |
| Precision |     0.6437 |
| Recall    |     0.7485 |
| ASSD      | 20.5254 px |
| Hausdorff | 85.8383 px |

Median boundary metrics:

| Metric    |     Median |
| --------- | ---------: |
| ASSD      | 18.3647 px |
| Hausdorff | 84.6753 px |

All 150 validation samples had valid prediction and target boundaries.

---

## 6. Correlation Analysis

Pearson correlations were calculated using the per-image validation results.

### Dice vs. ASSD

```text
r = -0.8144
```

This is a strong negative correlation.

Lower Dice values are generally associated with larger average boundary distances.

### IoU vs. ASSD

```text
r = -0.8090
```

IoU shows a similar relationship with ASSD.

### Dice vs. Hausdorff

```text
r = -0.4444
```

The relationship is substantially weaker than the Dice–ASSD relationship.

This difference is expected because Hausdorff distance is highly sensitive to a single extreme boundary discrepancy, whereas ASSD summarizes average boundary disagreement.

### ASSD vs. Hausdorff

```text
r = 0.6552
```

The two boundary metrics are positively related but capture different aspects of geometric error.

---

## 7. Boundary Failure Profile

Validation samples were classified using predefined operational thresholds.

| Profile                 |   Count | Percentage |
| ----------------------- | ------: | ---------: |
| Good                    |      40 |     26.67% |
| Broad boundary mismatch |      19 |     12.67% |
| FP-dominated            |      24 |     16.00% |
| Mixed / other           |      67 |     44.67% |
| **Total**               | **150** |   **100%** |

### Broad boundary mismatch

19 samples were classified as:

```text
Dice < 0.50
ASSD >= 30 px
```

Mean metrics:

| Metric    |        Mean |
| --------- | ----------: |
| Dice      |      0.2866 |
| Precision |      0.2223 |
| Recall    |      0.6334 |
| ASSD      |  44.6933 px |
| Hausdorff | 129.1036 px |

These samples combine poor overlap with substantial spatial boundary disagreement.

Visual inspection of representative cases also showed clear spatial misalignment and substantial false-positive prediction.

---

## 8. Mixed / Other Decomposition

The 67 samples initially classified as `mixed_other` were further decomposed using mutually exclusive operational rules.

| Subtype                 |  Count | % of Validation |
| ----------------------- | -----: | --------------: |
| Moderate boundary error |     49 |          32.67% |
| FN-heavy                |      8 |           5.33% |
| FP-heavy                |      2 |           1.33% |
| High-quality            |      5 |           3.33% |
| Severe mixed            |      2 |           1.33% |
| Hausdorff outlier       |      1 |           0.67% |
| **Total**               | **67** |      **44.67%** |

### Moderate boundary error

This was the dominant subtype within `mixed_other`.

49 samples were classified as:

```text
Dice >= 0.50
ASSD > 15 px
```

Mean metrics:

| Metric    |       Mean |
| --------- | ---------: |
| Dice      |     0.6777 |
| Precision |     0.7378 |
| Recall    |     0.6967 |
| ASSD      | 21.3632 px |
| Hausdorff | 97.3596 px |

These samples generally retain reasonable overlap and do not show the extreme precision/recall imbalance seen in the dedicated FP/FN profiles.

Their main distinguishing characteristic is increased boundary distance.

---

## 9. False-Positive and False-Negative Patterns

The analysis confirms that both FP-heavy and FN-heavy failures exist, but they occur at different frequencies.

### FP-dominated profile

24 / 150 validation samples:

```text
Precision < 0.50
Recall >= 0.50
```

Mean:

```text
Precision = 0.3758
Recall    = 0.8925
Dice      = 0.5181
```

This represents a high-recall / low-precision failure pattern.

### FN-heavy mixed subtype

8 / 150 validation samples:

```text
Recall < 0.60
Precision >= 0.60
```

Mean:

```text
Precision = 0.8658
Recall    = 0.3647
Dice      = 0.4965
```

This represents a conservative prediction pattern with substantial missed foreground.

---

## 10. Hausdorff Outlier Behavior

Visual and quantitative analysis identified cases where Dice and ASSD were relatively good but Hausdorff distance was extremely high.

For example:

```text
Dice      = 0.8908
ASSD      = 5.23 px
Hausdorff = 219.94 px
```

Another case showed:

```text
Dice      = 0.9615
ASSD      = 3.24 px
Hausdorff = 166.60 px
```

These examples demonstrate that Hausdorff distance can be dominated by a localized extreme boundary discrepancy even when overall segmentation quality is high.

Therefore Hausdorff should not be interpreted independently of ASSD and overlap metrics.

---

## 11. Relationship to Previous Experiments

Day 39 investigated Tversky Loss as a method for increasing the penalty on false negatives.

The experiment showed that increasing the FN penalty increased recall but substantially reduced precision and Dice.

Day 40 showed that the validation Dice curve was relatively flat around the selected threshold:

```text
0.50–0.60 threshold range
Dice variation ≈ 0.00053
```

Therefore small threshold changes are unlikely to completely resolve the observed failure patterns.

Day 41 investigated connected-component post-processing.

The locked validation-selected minimum component area was 200 pixels, but the locked test improvement over the previous baseline was only marginal:

```text
Dice:      +0.0007
IoU:       +0.0010
Precision: +0.0015
Recall:    -0.0012
```

This suggests that removing small isolated components alone is not sufficient to address the broader segmentation errors.

---

## 12. Visual Failure Findings

Representative boundary cases were inspected for:

* Worst Dice
* Worst ASSD
* High Dice with extreme Hausdorff
* Very high Dice with extreme Hausdorff
* Moderate Dice with high ASSD
* Representative median Dice

The visual inspection showed:

1. Severe low-Dice cases can contain substantial spatial misalignment.
2. Some difficult cases contain large amounts of false-positive prediction.
3. Moderate-Dice/high-ASSD cases can retain substantial target coverage while exhibiting poor spatial agreement.
4. High-Dice/extreme-Hausdorff cases demonstrate that a localized boundary outlier can produce a very large Hausdorff distance.

These observations are consistent with the quantitative boundary analysis.

---

## 13. Main Findings

### Finding 1 — Boundary disagreement is strongly associated with poor overlap

Dice and ASSD show a strong negative correlation:

```text
r = -0.8144
```

This indicates that average boundary disagreement is an important component of poor segmentation performance.

### Finding 2 — Moderate boundary error is more common than extreme failure

Using the predefined operational thresholds, 49 validation samples (32.67%) fall into the moderate boundary-error subtype.

This is substantially larger than the rare Hausdorff-outlier category.

### Finding 3 — False-positive failures are important but heterogeneous

A dedicated FP-dominated profile accounts for 16.00% of validation samples.

However, visual inspection and the mixed decomposition indicate that not all boundary errors are explained by false positives.

### Finding 4 — Under-segmentation exists but is not the dominant profile

The FN-heavy mixed subtype accounts for 5.33% of validation samples.

The Day 39 Tversky experiment also showed that aggressively increasing FN penalties can increase false positives and reduce overall Dice.

### Finding 5 — Hausdorff identifies rare extreme spatial errors

Hausdorff can become very large even when Dice and ASSD remain relatively good.

Therefore Hausdorff is useful as a complementary worst-case boundary metric rather than a standalone segmentation-quality metric.

---

## 14. Engineering Interpretation

The current evidence suggests that the next improvement effort should not focus exclusively on threshold tuning or increasing the false-negative penalty.

Instead, the failure analysis points toward improving:

* Spatial localization
* Boundary quality
* False-positive control
* Robustness on difficult visual patterns

The current results do not establish a single root cause.

The evidence supports a more cautious conclusion:

> Boundary disagreement is a substantial component of segmentation error, while false-positive prediction and spatial misalignment are important contributors to difficult validation cases.

---

## 15. Limitations

This analysis has several limitations.

### 15.1 Image-space distances

ASSD and Hausdorff are measured in pixels after resizing to 256 × 256.

They are not physical distances in millimeters.

### 15.2 Operational classification thresholds

The profile thresholds were defined for engineering analysis.

They are not clinical or universally standardized thresholds.

### 15.3 Boundary definition

The current implementation uses a one-pixel inner boundary obtained through binary erosion.

Different boundary definitions can produce different absolute boundary metrics.

### 15.4 Validation-only analysis

Failure profiles were derived from the validation set.

The test set was not used to tune the thresholds or failure classifications.

### 15.5 Correlation does not establish causation

The strong Dice–ASSD correlation demonstrates association, not that boundary error is the sole underlying cause of segmentation failure.

---

## 16. Outputs

### Code

```text
src/medseg/evaluation/boundary_metrics.py
scripts/analyze_boundary_metrics.py
scripts/visualize_boundary_cases.py
scripts/analyze_boundary_error_profile.py
scripts/decompose_mixed_boundary_errors.py
```

### Tests

```text
tests/test_boundary_metrics.py
```

Boundary metric tests:

```text
9 passed
```

Full project regression tests also passed after the boundary metric implementation.

### Reports

```text
artifacts/reports/day42_boundary_validation_per_image.csv
artifacts/reports/day42_boundary_validation.json
artifacts/reports/day42_boundary_error_profile.csv
artifacts/reports/day42_boundary_error_profile.json
artifacts/reports/day42_mixed_other_decomposition.csv
artifacts/reports/day42_mixed_other_decomposition.json
```

### Visualizations

```text
artifacts/figures/day42_boundary_cases/
```

---

## 17. Day 42 Conclusion

Day 42 established a boundary-aware evaluation layer for the medical segmentation pipeline.

The analysis shows that:

* Boundary disagreement is strongly associated with low Dice.
* Moderate boundary errors are common in the validation set.
* False-positive and false-negative failure modes coexist.
* Hausdorff captures rare extreme boundary outliers that Dice and ASSD can underrepresent.
* Connected-component filtering and threshold changes alone are unlikely to solve the broader failure distribution.

The most actionable direction for subsequent experimentation is therefore to investigate methods that improve **spatial localization and boundary quality while controlling false-positive prediction**, rather than simply increasing foreground sensitivity.

No model retraining or test-set optimization was performed during Day 42.
