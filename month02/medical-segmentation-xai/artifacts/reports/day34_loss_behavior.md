# Day 34 — Segmentation Loss Behavior

## Objective

Evaluate BCE, Dice, and BCE + Dice loss under controlled segmentation scenarios.

## Test Cases

1. Perfect foreground prediction
2. Completely wrong foreground prediction
3. Small foreground
4. Empty foreground mask

## Expected Behavior

### Perfect prediction

All losses should approach zero.

### Completely wrong prediction

BCE and Dice loss should increase substantially.

### Small foreground

The experiment demonstrates the different behavior of pixel-wise BCE and region-overlap-based Dice loss under foreground-background imbalance.

### Empty mask

Dice loss must remain finite and must not produce NaN.

## Gradient Validation

BCE + Dice loss supports backward propagation and produces finite gradients.

## Engineering Conclusion

BCE provides pixel-level supervision while Dice directly encourages foreground-region overlap.

Combining BCE and Dice provides a strong baseline for binary medical image segmentation, particularly when foreground regions are relatively small compared with the background.

Loss behavior was validated before beginning full model training.