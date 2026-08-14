# Day 7 — Neural Network Training with NumPy and PyTorch

## Overview

Day 7 focused on implementing and training a two-layer neural network from scratch with NumPy, then reproducing the same network using PyTorch.

The goal was to understand the complete neural network training process:

**Forward Pass → Loss → Backpropagation → Gradient Descent → Parameter Update**

---

## 1. Neural Network Architecture

The network used:

```text
Input
  ↓
Linear Layer (1 → 2)
  ↓
ReLU
  ↓
Linear Layer (2 → 1)
  ↓
Prediction