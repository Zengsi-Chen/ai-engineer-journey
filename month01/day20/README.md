# Day 20 - Classification Inference Pipeline & Testing

## Goal

Build a reusable classification inference pipeline and verify its behavior with automated tests.

## Inference Pipeline

The inference pipeline follows:

Raw Input
→ Preprocessing
→ Model Inference
→ Sigmoid
→ Probability
→ Best Threshold
→ Prediction
→ ClassificationResult

## ClassificationResult

Inference results are returned as a structured object:

- probability
- prediction

This makes the inference API easier to use and extend.

## Threshold

The classification decision uses:

prediction = probability >= threshold

The threshold was obtained from the previous classification evaluation work.

## Testing

The pipeline is tested with pytest.

Test coverage includes:

1. Batch output size
2. Probability range
3. Prediction values
4. Threshold logic
5. Deterministic inference
6. Invalid input shape
7. Threshold boundary cases
8. Real inference threshold behavior

## Boundary Testing

Special attention is given to:

probability < threshold → 0

probability = threshold → 1

probability > threshold → 1

## Running the Inference Demo

```cmd
py main.py

py -m pytest -v

Engineering Lessons
Separate inference logic from application code.
Return structured inference results.
Use pytest for automated testing.
Use fixtures for shared test resources.
Use parameterization for multiple test cases.
Test both valid and invalid inputs.
Test boundary conditions explicitly.
Regression tests should execute the real production code path.