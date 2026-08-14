# Day 6: Gradient Descent from Scratch

## Goal

Understand how a machine learning model learns by implementing gradient descent from scratch using NumPy.

## What I Learned

* Prediction
* Mean Squared Error (MSE)
* Loss
* Gradient
* Learning Rate
* Gradient Descent
* Model parameters: `w` and `b`
* Training loop
* Basic AI code organization with functions

## Model

The model used in this project is:

```text
y = wx + b
```

The training data follows:

```text
x = [1, 2, 3, 4]
y = [3, 5, 7, 9]
```

The target relationship is:

```text
y = 2x + 1
```

The model starts with:

```text
w = 0
b = 0
```

and learns the parameters through gradient descent.

## Training Process

The training loop follows:

```text
Input
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Update Parameters
  ↓
Repeat
```

Prediction:

```text
prediction = wx + b
```

Loss:

```text
Loss = mean((prediction - y)²)
```

Parameter updates:

```text
w = w - learning_rate × gradient_w
b = b - learning_rate × gradient_b
```

## Learning Rate Experiment

I tested different learning rates:

```text
0.001 → learning is slow
0.1   → stable and effective
1.0   → can become unstable
```

This demonstrated why learning rate is an important hyperparameter in machine learning.

## Code Structure

The training program is organized into separate functions:

```text
predict()
compute_loss()
compute_gradients()
train()
main()
```

This introduces basic separation of concerns and prepares for larger AI engineering projects.

## Result

After training, the model learns approximately:

```text
w ≈ 2
b ≈ 1
```

Therefore:

```text
y ≈ 2x + 1
```

The model successfully learned the underlying relationship from the training data.

## Key Takeaway

Gradient descent allows a model to learn by repeatedly:

```text
Predict → Calculate Loss → Calculate Gradient → Update Parameters
```

This is the fundamental idea behind training modern machine learning and neural network models.

## Environment

* Python
* NumPy
* Git
* GitHub

## Commit

```text
Day 6: Implement gradient descent from scratch
```
