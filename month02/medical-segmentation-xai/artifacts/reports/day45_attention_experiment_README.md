# Day 45 — Attention U-Net Architecture Experiment

## 1. Objective

Evaluate whether attention-gated skip connections can reduce false-positive activation and improve spatial localization without sacrificing overall segmentation performance.

The experiment was motivated by the failure analysis accumulated during Days 39–44.

The baseline U-Net showed relatively high recall but substantially lower precision:

* Validation Dice: 0.6366
* Validation Precision: 0.6437
* Validation Recall: 0.7485
* Validation ASSD: 20.5254
* Validation Hausdorff: 85.8383

Previous experiments suggested that increasing foreground sensitivity could worsen false-positive activation.

Therefore, Day 45 tested whether an architectural mechanism could selectively suppress irrelevant encoder features during skip-connection fusion.

---

## 2. Research Question

> Can attention-gated skip connections reduce false-positive activation and improve spatial localization without sacrificing overall segmentation performance?

---

## 3. Hypothesis

### H1

Attention-gated skip connections will suppress irrelevant background features passed through the encoder-decoder skip connections.

Expected effects:

* Precision ↑
* Dice ↑
* IoU ↑
* ASSD ↓
* Hausdorff ↓

while maintaining acceptable recall.

### Alternative outcome

If Attention U-Net increases recall but decreases precision, this would indicate that the architecture is encouraging broader foreground activation rather than solving the dominant false-positive problem.

---

## 4. Experimental Design

A controlled A/B architecture experiment was used.

### Baseline

Standard U-Net with BCE + Dice loss.

### Experimental model

Attention U-Net with attention-gated skip connections.

### Fixed variables

| Variable                  | Configuration     |
| ------------------------- | ----------------- |
| Dataset                   | Kvasir-SEG        |
| Train / Validation / Test | 700 / 150 / 150   |
| Image size                | 256 × 256         |
| Augmentation              | None              |
| Base architecture         | U-Net family      |
| Features                  | (16, 32, 64, 128) |
| Loss                      | BCE + Dice        |
| BCE weight                | 0.5               |
| Dice weight               | 0.5               |
| Optimizer                 | AdamW             |
| Learning rate             | 1e-3              |
| Weight decay              | 1e-4              |
| Batch size                | 4                 |
| Epochs                    | 10                |
| Random seed               | 42                |
| Device                    | CPU               |
| Threshold                 | 0.55              |
| Post-processing           | None              |

The threshold of 0.55 was locked from the Day 40 validation threshold analysis.

No test-set tuning was performed.

---

## 5. Attention U-Net Architecture

The Attention U-Net adds an attention gate before each skip connection is concatenated with the decoder feature map.

The attention gate receives:

* encoder skip features
* decoder gating features

and produces an attention map used to modulate the encoder features.

The implementation consists of:

* `AttentionGate`
* `AttentionDecoderBlock`
* `AttentionUNet`

The original `UNet` implementation was not modified.

A separate model implementation was added:

```text
src/medseg/models/attention_unet.py
```

The model was integrated through the model factory:

```text
src/medseg/models/factory.py
```

using:

```text
attention_unet
```

as the model name.

---

## 6. Parameter Comparison

Using the production experiment configuration `(16, 32, 64, 128)`:

| Model           | Parameters |
| --------------- | ---------: |
| Baseline U-Net  |  1,942,577 |
| Attention U-Net |  1,964,941 |

Additional parameters:

```text
22,364
```

Relative increase:

```text
approximately 1.15%
```

Therefore, the experiment represents a relatively small increase in model capacity.

---

## 7. Implementation Validation

The Attention U-Net implementation was validated before training.

### Forward-pass test

Input:

```text
[2, 3, 256, 256]
```

Output:

```text
[2, 1, 256, 256]
```

### Automated tests

Three Attention U-Net tests were added:

* output shape
* factory integration
* parameter count

Result:

```text
3 passed
```

---

## 8. Training Results

The Attention U-Net was trained for 10 epochs.

| Epoch | Train Loss | Val Loss |     Val Dice |      Val IoU |
| ----: | ---------: | -------: | -----------: | -----------: |
|     1 |   0.618753 | 0.512490 |     0.424036 |     0.301628 |
|     2 |   0.516011 | 0.463892 |     0.479627 |     0.349089 |
|     3 |   0.469049 | 0.446905 |     0.455235 |     0.334415 |
|     4 |   0.448104 | 0.417430 |     0.532167 |     0.400381 |
|     5 |   0.415933 | 0.401260 |     0.566046 |     0.432146 |
|     6 |   0.406877 | 0.380116 |     0.588553 |     0.462764 |
|     7 |   0.379985 | 0.374547 |     0.615241 |     0.482817 |
|     8 |   0.363159 | 0.390888 | **0.631770** | **0.495693** |
|     9 |   0.347184 | 0.366197 |     0.602522 |     0.484360 |
|    10 |   0.333380 | 0.438665 |     0.494386 |     0.383411 |

Best checkpoint:

```text
Epoch 8
```

Best validation Dice:

```text
0.631770
```

Best validation IoU:

```text
0.495693
```

The later decline in Dice after epoch 8 demonstrates why the best validation checkpoint, rather than the final epoch, was used.

---

## 9. Validation Results

The locked validation threshold was 0.55.

### Baseline vs Attention U-Net

| Metric      | Baseline U-Net | Attention U-Net |  Change |
| ----------- | -------------: | --------------: | ------: |
| Dice ↑      |         0.6366 |          0.6318 | -0.0049 |
| IoU ↑       |         0.4996 |          0.4957 | -0.0039 |
| Precision ↑ |         0.6437 |          0.5934 | -0.0503 |
| Recall ↑    |         0.7485 |          0.8182 | +0.0696 |
| ASSD ↓      |        20.5254 |         22.0659 | +1.5405 |
| Hausdorff ↓ |        85.8383 |         87.3973 | +1.5590 |

Attention U-Net did not improve the primary overlap metrics.

More importantly, precision decreased substantially while recall increased.

Boundary metrics also moved in the wrong direction on average.

---

## 10. Paired Validation Audit

The two models were evaluated on exactly the same 150 validation samples.

Paired sample IDs:

```text
150 / 150 matched
```

### Paired metric changes

| Metric    | Mean Change | Median Change |
| --------- | ----------: | ------------: |
| Dice      |     -0.0049 |       -0.0056 |
| IoU       |     -0.0039 |       -0.0062 |
| Precision |     -0.0503 |       -0.0414 |
| Recall    |     +0.0696 |       +0.0445 |
| ASSD      |     +1.5405 |       +1.0519 |
| Hausdorff |     +1.5590 |       -0.2369 |

### Improvement counts

| Metric    | Improved | Worsened | Unchanged |
| --------- | -------: | -------: | --------: |
| Dice      |       70 |       80 |         0 |
| IoU       |       70 |       80 |         0 |
| Precision |       41 |      109 |         0 |
| Recall    |      115 |       34 |         1 |
| ASSD      |       67 |       83 |         0 |
| Hausdorff |       76 |       74 |         0 |

The paired analysis provides stronger evidence than the aggregate means.

Attention U-Net improved recall in 115/150 samples, but precision worsened in 109/150 samples.

---

## 11. Precision–Recall Trade-off

The validation results reveal a consistent behavioral change.

### Recall

```text
0.7485 → 0.8182
```

Change:

```text
+0.0696
```

Recall improved in:

```text
115 / 150 samples
```

### Precision

```text
0.6437 → 0.5934
```

Change:

```text
-0.0503
```

Precision worsened in:

```text
109 / 150 samples
```

This indicates that Attention U-Net tended to activate
