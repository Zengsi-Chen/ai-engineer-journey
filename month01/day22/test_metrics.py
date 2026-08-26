import torch

from metrics import RegressionMetrics


metrics = RegressionMetrics()

predictions = torch.tensor([
    [1.0],
    [3.0],
    [5.0]
])

targets = torch.tensor([
    [1.0],
    [4.0],
    [7.0]
])

result = metrics.compute(
    predictions,
    targets
)

print(result)