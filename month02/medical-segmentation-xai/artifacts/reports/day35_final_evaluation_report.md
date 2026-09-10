① Experiment Configuration

Dataset: Kvasir-SEG
Total samples: 1000

Train: 700
Validation: 150
Test: 150

Image size: 256 × 256

Model: U-Net
Features: (16, 32, 64, 128)

Parameters: 1,942,577

Device: CPU
Seed: 42

② Training configuration

Loss:
0.5 × BCE + 0.5 × Dice

Optimizer:
AdamW

Learning rate:
1e-3

Weight decay:
1e-4

Scheduler:
ReduceLROnPlateau

Early stopping:
enabled

③ Threshold selection

Threshold candidates:
0.10 → 0.90

Validation best threshold:
0.55

Test threshold:
0.55 (LOCKED)

④ Validation results

Dice       0.6366
IoU        0.4995
Precision  0.6437
Recall     0.7485

⑤ Final Test results
Dice       0.6144
IoU        0.4812
Precision  0.5982
Recall     0.7654
⑥ Generalization gap
Dice gap:
-0.0222

IoU gap:
-0.0183

Precision gap:
-0.0455

Recall gap:
+0.0169
⑦ Engineering interpretation

The model demonstrates reasonable generalization from validation to the independent test set, with a Dice score of 0.6144 and an absolute validation-to-test Dice gap of 0.0222. The model maintains relatively high recall (0.7654) but lower precision (0.5982), indicating a tendency toward over-segmentation and false-positive predictions. The threshold of 0.55 was selected exclusively on the validation set and remained locked during final test evaluation.
