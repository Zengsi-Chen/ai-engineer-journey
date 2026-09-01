# Day 26 — Transfer Learning Deep Dive

## Fine-Tuning Strategy & Domain Shift

Day 26 focuses on a central question in Transfer Learning:

> When a trained model is applied to a shifted target domain, how should we fine-tune it?

The project uses the ResNet-18 model trained previously on CIFAR-10 and evaluates its robustness under different domain shifts.

---

# Learning Objectives

By the end of Day 26, I learned how to:

* Analyze domain shift
* Measure model performance degradation
* Build synthetic target domains
* Implement Feature Extraction
* Implement Partial Fine-Tuning
* Implement Full Fine-Tuning
* Freeze and unfreeze different parts of a neural network
* Compare Fine-Tuning strategies
* Measure training time
* Measure accuracy gain
* Analyze accuracy vs training-time trade-offs
* Make an engineering decision based on experiment results

---

# Project Structure

```text
day26/
│
├── main.py
├── model.py
├── resnet.py
│
├── domain_shift.py
├── domain_experiment.py
├── target_domain_data.py
│
├── finetuning_strategy.py
├── finetuning_pipeline.py
│
├── strategy_comparison.py
│
├── tests/
│   ├── test_domain_shift.py
│   ├── test_finetuning_strategy.py
│   └── test_strategy_comparison.py
│
└── artifacts/
    └── checkpoints/
        └── best_resnet18.pth
```

---

# 1. Domain Shift

A model can perform well on the source domain but lose accuracy when the input distribution changes.

```text
Source Domain
    ↓
Day 25 CIFAR-10 Training
    ↓
Trained ResNet-18
    ↓
Target Domain
    ↓
Different Brightness
Different Contrast
Grayscale
Noise
    ↓
Performance Drop
```

The synthetic target domain was created using:

* Brightness shift
* Contrast shift
* Grayscale conversion
* Gaussian noise

The configuration is represented by:

```python
@dataclass
class DomainShiftConfig:
    brightness: float = 0.0
    contrast: float = 0.0
    grayscale: bool = False
    noise_std: float = 0.0
```

---

# 2. Baseline Domain Shift Experiment

The trained Day 25 ResNet-18 model was evaluated without additional fine-tuning.

## Results

| Domain         | Accuracy | Correct Predictions |
| -------------- | -------: | ------------------: |
| No Shift       |   34.28% |        3428 / 10000 |
| Moderate Shift |   32.05% |        3205 / 10000 |
| Severe Shift   |   17.67% |        1767 / 10000 |

## Accuracy Drop

```text
No Shift
34.28%

        ↓

Moderate Shift
32.05%

Accuracy Drop
2.23 percentage points
```

The severe shift produced a much larger performance degradation:

```text
34.28%
   ↓
17.67%

Drop: 16.61 percentage points
```

This demonstrates that:

> A model's performance on its training domain does not guarantee the same performance on a shifted target domain.

---

# 3. Target Domain

The target domain used for Fine-Tuning was the moderate domain shift.

```python
target_config = DomainShiftConfig(
    brightness=0.2,
    contrast=0.2,
    grayscale=False,
    noise_std=0.02,
)
```

This creates a target distribution that differs from the original CIFAR-10 training distribution.

---

# 4. Fine-Tuning Strategies

Three strategies were implemented.

## Strategy 1 — Feature Extraction

Only the classifier is trainable.

```text
Frozen
├── stem
├── layer1
├── layer2
├── layer3
└── layer4

Trainable
└── classifier
```

This is the cheapest strategy because most of the network remains frozen.

---

## Strategy 2 — Partial Fine-Tuning

Only the final ResNet block and classifier are trainable.

```text
Frozen
├── stem
├── layer1
├── layer2
└── layer3

Trainable
├── layer4
└── classifier
```

This allows high-level features to adapt to the target domain while preserving lower-level features.

---

## Strategy 3 — Full Fine-Tuning

The entire model is trainable.

```text
Trainable
├── stem
├── layer1
├── layer2
├── layer3
├── layer4
└── classifier
```

This provides the maximum adaptation capacity but requires the most computation.

---

# 5. Fine-Tuning Implementation

The strategy system first freezes the entire model:

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

It then unfreezes only the selected layers:

```python
for layer_name in strategy.trainable_layers:

    layer = getattr(
        model,
        layer_name,
    )

    for parameter in layer.parameters():
        parameter.requires_grad = True
```

This makes the fine-tuning behavior explicit and easy to experiment with.

---

# 6. Experiment Fairness

Each experiment:

1. Reloaded the same Day 25 checkpoint
2. Used the same moderate target domain
3. Used the same batch size
4. Used the same number of epochs
5. Started from the same model state

```text
Day 25 Checkpoint
       │
       ├── Feature Extraction
       │
       ├── Partial Fine-Tuning
       │
       └── Full Fine-Tuning
```

Reloading the checkpoint before every experiment prevents one strategy from continuing the training of another.

---

# 7. Experiment Metrics

The following metrics were recorded:

* Best Accuracy
* Best Epoch
* Final Accuracy
* Training Time
* Accuracy Gain
* Accuracy Gain per Second

Accuracy gain is defined as:

```text
Accuracy Gain
=
Fine-Tuned Accuracy
-
Moderate Shift Baseline
```

The moderate-shift baseline was:

```text
0.3205
```

Therefore:

```text
Accuracy Gain
=
Best Fine-Tuned Accuracy
-
0.3205
```

Efficiency is defined as:

```text
Accuracy Gain
────────────────────
Training Time
```

This measures the improvement obtained per second of training.

---

# 8. Fine-Tuning Results

> Replace the values below with the actual results produced by `py main.py`.

| Strategy            | Best Epoch | Best Accuracy | Final Accuracy | Training Time | Accuracy Gain |
| ------------------- | ---------: | ------------: | -------------: | ------------: | ------------: |
| Feature Extraction  |       TODO |          TODO |           TODO |        TODO s |          TODO |
| Partial Fine-Tuning |       4 |          0.6709 |            0.6637 |        62937.49 s |          0.3504 |
| Full Fine-Tuning    |       TODO |          TODO |           TODO |        TODO s |          TODO |

---

# 9. Accuracy vs Training Time

The highest accuracy does not automatically mean the best engineering strategy.

The decision must consider:

```text
Accuracy
    +
Accuracy Gain
    +
Training Time
    =
Engineering Trade-Off
```

Three possible outcomes are important.

## Highest Accuracy

The strategy with the largest:

```text
Best Accuracy
```

is found using:

```python
find_best_strategy()
```

---

## Most Efficient

The strategy with the largest:

```text
Accuracy Gain / Training Time
```

is found using:

```python
find_most_efficient_strategy()
```

---

## Best Overall Trade-Off

The best engineering choice may be neither:

```text
Highest Accuracy
```

nor:

```text
Shortest Training Time
```

Instead, it may be the strategy that provides:

```text
Strong Accuracy
+
Reasonable Training Time
```

This is often where Partial Fine-Tuning becomes useful.

---

# 10. Engineering Conclusion

The domain shift experiment showed that the Day 25 ResNet-18 model was sensitive to changes in the input distribution.

Performance decreased from:

```text
No Shift
34.28%
```

to:

```text
Moderate Shift
32.05%
```

and further to:

```text
Severe Shift
17.67%
```

This confirms that model performance should not be evaluated only on the original source domain.

Fine-Tuning provides a way to adapt the trained model to the target domain.

The three strategies represent different levels of adaptation:

```text
Feature Extraction
        ↓
Lowest computational cost

Partial Fine-Tuning
        ↓
Balanced adaptation and cost

Full Fine-Tuning
        ↓
Maximum adaptation capacity
        +
Highest computational cost
```

The final strategy selection should be based on the actual experiment results.

If Full Fine-Tuning provides only a small improvement but requires substantially more training time, it may not be the best engineering choice.

If Partial Fine-Tuning achieves nearly the same accuracy with lower computational cost, it may provide the best trade-off.

If Feature Extraction provides strong improvement with very low training cost, it may be the most efficient solution.

The main engineering lesson is:

> The best model is not always the model with the highest accuracy.

A real deployment decision should consider:

```text
Accuracy
+
Adaptation Quality
+
Training Cost
+
Compute Resources
+
Time Constraints
```

---

# 11. Key Lessons

## Domain Shift

```text
Training Accuracy
≠
Target Domain Accuracy
```

---

## Fine-Tuning Depth

More trainable layers provide more adaptation capacity:

```text
Classifier Only
    ↓
Layer 4 + Classifier
    ↓
Entire Network
```

But this also increases computational cost.

---

## Fair Experiment Design

Every strategy should start from:

```text
Same Checkpoint
+
Same Target Domain
+
Same Training Conditions
```

Otherwise, the comparison is not reliable.

---

## Accuracy Is Not Enough

A good experiment compares:

```text
Best Accuracy
Accuracy Gain
Training Time
Accuracy per Second
```

---

# Final Day 26 Conclusion

Day 26 transformed Transfer Learning from a simple workflow:

```text
Load Model
    ↓
Fine-Tune
```

into an engineering experiment:

```text
Domain Shift
    ↓
Measure Baseline
    ↓
Create Target Domain
    ↓
Select Fine-Tuning Strategies
    ↓
Train Independent Experiments
    ↓
Measure Accuracy
    ↓
Measure Training Cost
    ↓
Compare Results
    ↓
Make an Engineering Decision
```

The most important lesson from Day 26 is:

> Fine-Tuning is a trade-off between adaptation capacity, accuracy, and computational cost.

---

# How to Run

Activate the virtual environment and run:

```cmd
cd C:\Users\ABC\Documents\ai-engineer-journey\month01\day26

py main.py
```

Run the tests:

```cmd
py -m pytest tests\ -v
```

---

# Day 26 Status

```text
✓ Domain Shift
✓ Brightness Shift
✓ Contrast Shift
✓ Grayscale Shift
✓ Gaussian Noise
✓ Target Domain Data
✓ Feature Extraction
✓ Partial Fine-Tuning
✓ Full Fine-Tuning
✓ Fine-Tuning Strategy Tests
✓ Accuracy Gain
✓ Training Time Measurement
✓ Accuracy / Training Time Trade-Off
✓ Strategy Comparison
✓ Engineering Decision Framework
```

## Day 26 Complete

The next stage is to move from manually defined domain shifts and fine-tuning experiments toward more realistic computer vision transfer learning workflows.
