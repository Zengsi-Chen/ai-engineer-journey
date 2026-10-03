# Day 43 — Boundary-Aware Improvement Experiment Design

## 1. Objective

Design a controlled experiment to determine whether a boundary-aware training objective can improve segmentation boundary quality without substantially degrading overall segmentation performance.

This experiment is motivated by the Day 42 boundary analysis, which showed a strong negative correlation between Dice and Average Symmetric Surface Distance (ASSD).

---

## 2. Research Question

**RQ43:**

Does adding a boundary-aware loss to the current U-Net training objective improve boundary quality while maintaining or improving overall segmentation performance?

---

## 3. Motivation from Day 42

Day 42 established a boundary-aware evaluation layer using:

* Dice
* IoU
* Precision
* Recall
* Average Symmetric Surface Distance (ASSD)
* Hausdorff distance

The validation analysis showed:

* Dice vs ASSD Pearson correlation: **-0.8144**
* IoU vs ASSD Pearson correlation: **-0.8090**
* Moderate boundary errors were common.
* False-positive prediction and spatial misalignment were important contributors to difficult cases.
* Hausdorff distance identified localized extreme boundary errors that were not always reflected by Dice or ASSD.

These findings motivate testing a training objective that explicitly considers boundary quality.

---

## 4. Experimental Hypothesis

### Primary hypothesis

Adding a boundary-aware component to the training loss will reduce boundary error, measured by ASSD and Hausdorff distance.

### Secondary hypothesis

A useful boundary-aware objective should improve or preserve Dice and IoU without causing a large degradation in precision.

### Null hypothesis

Adding a boundary-aware objective does not produce a meaningful improvement in boundary quality or overall segmentation quality.

---

## 5. Experimental Groups

### Experiment A — Baseline

Current training objective:

**BCE + Dice**

This is the control condition.

### Experiment B — Boundary-Aware Combined Loss

Training objective:

**BCE + Dice + Boundary Loss**

The boundary component will be introduced while preserving the existing BCE + Dice objective.

### Experiment C — Boundary-Emphasis Loss

Training objective:

**Dice + Boundary Loss**

This experiment evaluates whether the boundary component can remain effective without BCE.

The exact boundary-loss implementation and weighting will be specified before training and kept fixed throughout the experiment.

---

## 6. Controlled Variables

The following variables must remain identical across all experiments:

| Variable                  | Fixed value       |
| ------------------------- | ----------------- |
| Dataset                   | Kvasir-SEG        |
| Train / Validation / Test | 700 / 150 / 150   |
| Model                     | U-Net             |
| Features                  | (16, 32, 64, 128) |
| Input size                | 256 × 256         |
| Augmentation              | None              |
| Optimizer                 | AdamW             |
| Learning rate             | 1e-3              |
| Weight decay              | 1e-4              |
| Batch size                | 4                 |
| Epochs                    | 10                |
| Random seed               | 42                |
| Device                    | CPU               |
| Decision threshold        | 0.55              |
| Post-processing           | None              |

Only the training loss is allowed to change between experimental groups.

---

## 7. Independent Variable

The independent variable is the training loss formulation.

The experiment must not simultaneously change:

* architecture
* dataset split
* optimizer
* learning rate
* augmentation
* threshold
* post-processing

This isolates the effect of the training objective.

---

## 8. Evaluation Metrics

### Primary metrics

Higher is better:

* Dice
* IoU

Lower is better:

* ASSD
* Hausdorff distance

### Secondary metrics

Higher is better:

* Precision
* Recall

Additional prediction-behavior metric:

* Prediction foreground ratio

The target foreground ratio will also be reported for comparison.

---

## 9. Validation Protocol

Validation is used for experiment comparison and model selection.

For each experiment:

1. Train using the fixed training split.
2. Evaluate on the validation split.
3. Select the best checkpoint according to validation Dice.
4. Record all segmentation and boundary metrics.
5. Do not modify the test-set protocol based on test results.

The threshold remains fixed at **0.55**, based on the previously established validation operating point.

---

## 10. Locked Test Protocol

After validation selection:

1. Select the final checkpoint using validation results only.
2. Keep threshold fixed at 0.55.
3. Do not tune the loss weight using the test set.
4. Do not tune post-processing using the test set.
5. Evaluate the selected model once on the locked test set.

This prevents test-set optimization and preserves the integrity of the comparison.

---

## 11. Success Criteria

A boundary-aware experiment will be considered useful if it demonstrates an improvement in boundary quality while maintaining acceptable overall segmentation quality.

A strong outcome would show:

* Dice improves or remains approximately stable.
* IoU improves or remains approximately stable.
* ASSD decreases.
* Hausdorff decreases.
* Precision does not show a substantial degradation.
* Prediction foreground ratio does not increase abnormally.

A trade-off result must also be reported explicitly.

For example:

* lower ASSD but lower Dice
* lower Hausdorff but lower precision
* higher recall but substantially higher false-positive rate

Such outcomes are not automatically considered successful or unsuccessful; they describe the cost of the boundary-aware objective.

---

## 12. Expected Failure Modes

The experiment must explicitly monitor for:

### Failure Mode 1 — False-positive expansion

The model may predict excessive foreground in an attempt to cover boundaries.

Indicators:

* Precision decreases.
* Recall increases strongly.
* Prediction foreground ratio increases.
* Dice may decrease.

This failure mode is particularly important because a similar pattern occurred in the Tversky experiments.

### Failure Mode 2 — Boundary improvement without overlap improvement

ASSD may decrease while Dice and IoU remain unchanged or decrease.

This would indicate improved contour alignment without improved region-level segmentation.

### Failure Mode 3 — No meaningful effect

All metrics remain close to baseline.

This would indicate that the chosen boundary-aware objective does not materially change the current model's behavior under this configuration.

### Failure Mode 4 — Localized Hausdorff improvement only

Hausdorff may improve substantially while ASSD changes little.

This would indicate that the loss may primarily reduce extreme boundary outliers rather than improving the average boundary.

---

## 13. Comparison Table

The final experiment report will use the following structure:

| Experiment | Loss                  | Dice | IoU | Precision | Recall | ASSD | Hausdorff | FG Ratio |
| ---------- | --------------------- | ---: | --: | --------: | -----: | ---: | --------: | -------: |
| Baseline   | BCE + Dice            |    — |   — |         — |      — |    — |         — |        — |
| Boundary-A | BCE + Dice + Boundary |    — |   — |         — |      — |    — |         — |        — |
| Boundary-B | Dice + Boundary       |    — |   — |         — |      — |    — |         — |        — |

Changes relative to baseline will also be reported:

* Δ Dice
* Δ IoU
* Δ Precision
* Δ Recall
* Δ ASSD
* Δ Hausdorff
* Δ Prediction FG Ratio

---

## 14. Reproducibility Requirements

Every experiment must record:

* random seed
* model architecture
* loss configuration
* loss weights
* optimizer
* learning rate
* weight decay
* batch size
* number of epochs
* dataset split
* checkpoint selection rule
* threshold
* evaluation metrics

The exact configuration used to produce the final result must be saved as an artifact.

---

## 15. Relationship to Previous Experiments

### Day 39 — Tversky Loss

Tversky Loss increased recall but produced substantially more foreground prediction and lower overall segmentation quality.

Day 43 therefore treats false-positive control as an important constraint.

### Day 40 — Threshold Robustness

Threshold 0.55 was selected on validation and showed a relatively stable operating region.

Day 43 therefore keeps threshold 0.55 fixed.

### Day 41 — Connected-Component Post-processing

Connected-component filtering produced only marginal improvement on the locked test set.

Day 43 therefore focuses on changing the training objective rather than adding another inference-time modification.

### Day 42 — Boundary Analysis

Boundary error was strongly associated with low Dice.

Day 43 therefore tests whether boundary-aware training can directly address this error component.

---

## 16. Experimental Principle

The experiment follows:

**Error Analysis → Hypothesis → Controlled Variable → Training Experiment → Validation → Locked Test → Error Analysis**

The objective is not simply to maximize one metric.

The objective is to determine whether the new training objective produces a better overall error profile.

---

## 17. Current Status

**Day 43 Step 2 — Experiment design documented.**

No model training has been performed yet.

No test-set optimization has been performed.

The next step is to inspect the existing loss implementation and determine the most appropriate boundary-loss formulation that can be added without introducing unnecessary dependencies.
