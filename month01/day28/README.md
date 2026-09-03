Day 28 — Data Leakage & Robust Validation
Overview

Day 28 focuses on building reliable machine learning validation pipelines by preventing data leakage and correctly handling correlated samples.

The main goal is to move from simply obtaining a high validation score to building an evaluation process that can be trusted.

Learning Objectives

By the end of Day 28, the pipeline can:

Detect train / validation / test overlap.
Prevent preprocessing leakage.
Prevent normalization leakage.
Apply augmentation only to training data.
Perform sample-level dataset splitting.
Perform group-level dataset splitting.
Use K-Fold validation.
Use Stratified K-Fold validation.
Use Group K-Fold validation.
Automatically detect group leakage.
Integrate leakage protection into an image pipeline.
Test leakage prevention automatically.
Compare random splitting against group-aware splitting.
Key Concepts
1. Data Leakage

Data leakage occurs when information from validation or test data becomes available to the training process.

Leakage can produce artificially optimistic validation results and lead to incorrect model selection.

2. Train / Validation / Test

The roles of the three datasets are separated:

Train
  ↓
Model fitting
Preprocessing fitting
Normalization fitting
Training augmentation

Validation
  ↓
Model selection
Hyperparameter tuning
Early stopping

Test
  ↓
Final evaluation

The test set should remain isolated until final evaluation.

3. Common Leakage Types

The following leakage categories were studied:

Direct leakage
Preprocessing leakage
Normalization leakage
Augmentation leakage
Duplicate leakage
Group leakage

Group leakage is particularly important when multiple samples originate from the same patient, subject, video, session, or other logical entity.

4. Leakage Detection

Implemented:

src/leakage_checks.py

The detector checks:

Train ∩ Validation
Train ∩ Test
Validation ∩ Test

Any non-empty intersection is treated as leakage.

5. Safe Preprocessing

Implemented:

src/preprocessing.py

The SafePreprocessor requires fitting before transformation.

Correct workflow:

X_train
   ↓
fit()
   ↓
learn preprocessing parameters
   ↓
transform()
   ├── X_train
   ├── X_validation
   └── X_test
6. Train-Only Normalization

Implemented:

src/normalization.py

Normalization statistics are learned only from training images.

normalizer.fit(train_images)

train_images = normalizer.transform(train_images)
val_images = normalizer.transform(val_images)
test_images = normalizer.transform(test_images)

Validation and test statistics are never used to fit the normalizer.

7. Train-Only Augmentation

Implemented:

src/augmentation.py

Behavior:

Train       → Augmentation
Validation  → No Augmentation
Test        → No Augmentation

This keeps evaluation deterministic and representative of real inference data.

8. Sample vs Group Splitting

The pipeline supports two split units.

Sample-Level
split_unit="sample"

Used when individual samples can reasonably be considered independent.

Group-Level
split_unit="group"

Used when multiple samples belong to the same logical entity.

Examples:

Patient
Subject
Video
Session
Acquisition
Product

A group must never appear in multiple dataset partitions.

9. Cross Validation

Implemented validation strategies include:

K-Fold

Standard sample-level K-Fold validation.

Stratified K-Fold

Preserves class distribution across folds as much as possible.

Group K-Fold

Keeps all samples from the same group together.

The correct validation strategy depends on the dependency structure of the dataset.

10. Leakage-Safe Pipeline

Implemented:

src/leakage_safe_pipeline.py

The pipeline supports:

split_unit="sample"

and:

split_unit="group"

The complete workflow is:

Raw Dataset
     ↓
Train / Validation / Test Split
     ↓
Leakage Check
     ↓
Fit Normalizer on Training Data
     ↓
Transform All Splits
     ↓
Augment Training Data Only
     ↓
Prepared Dataset

For group splitting:

Group Overlap
     ↓
Detected
     ↓
Raise RuntimeError
11. Controlled Leakage Experiment

A synthetic experiment compared random sample-level splitting with group-aware splitting.

Dataset
Samples:            500
Groups:             100
Samples per group:  5
Features:           2
Results
Split Strategy	Validation Accuracy	Group Overlap
Random Split	0.6667	80
Group Split	0.5800	0

Difference:

0.6667 - 0.5800 = 0.0867

or:

+8.67 percentage points
Interpretation

The random split contained substantial group overlap, while the group-aware split contained no group overlap.

In this controlled synthetic experiment, the random split produced a validation accuracy 8.67 percentage points higher than the leakage-safe group split.

This demonstrates how correlated samples crossing dataset boundaries can make validation metrics optimistic.

The numerical difference should not be interpreted as a universal effect size for all leakage scenarios.

12. Automated Testing

Day 28 includes automated tests for:

Split overlap detection
Train/validation leakage
Train/test leakage
Validation/test leakage
Group overlap
Group split integrity
Safe preprocessing
Training-only normalization
Training-only augmentation
Deterministic validation transformation
Sample pipeline
Group pipeline
Missing group validation
Invalid configuration

Run:

py -m pytest tests -v

Expected result:

All tests PASSED
13. Engineering Principles

The most important principles from Day 28 are:

Split First

Dataset splitting must happen before fitting data-dependent transformations.

Fit on Train Only

Preprocessing and normalization parameters must be learned only from training data.

Augment Train Only

Validation and test data should not receive training-time augmentation.

Respect Data Dependencies

Samples from the same patient, subject, video, session, or entity should remain in the same split when appropriate.

Automate Leakage Detection

Do not rely only on manual inspection.

Protect the Test Set

The test set should not be repeatedly used for model selection.

14. Project Structure
day28/
│
├── artifacts/
│   └── reports/
│       └── day28_experiment_report.md
│
├── experiments/
│   ├── ...
│   └── compare_leakage_experiments_v4.py
│
├── src/
│   ├── augmentation.py
│   ├── cross_validation.py
│   ├── data_split.py
│   ├── leakage_checks.py
│   ├── leakage_safe_pipeline.py
│   ├── normalization.py
│   └── preprocessing.py
│
└── tests/
    ├── test_augmentation.py
    ├── test_group_split.py
    ├── test_leakage.py
    ├── test_leakage_safe_pipeline.py
    ├── test_normalization.py
    └── test_preprocessing.py
15. Final Takeaway

A machine learning metric is only meaningful when the evaluation pipeline is trustworthy.

The Day 28 pipeline therefore follows:

Correct Split
     ↓
No Leakage
     ↓
Training-Only Preprocessing
     ↓
Training-Only Normalization
     ↓
Training-Only Augmentation
     ↓
Reliable Validation
     ↓
Final Test

Day 28 establishes the foundation for reliable experimentation and prepares the project for more advanced model development.