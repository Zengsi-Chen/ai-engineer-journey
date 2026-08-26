import torch

from tasks import RegressionTask


task = RegressionTask()

outputs = torch.tensor([
    [1.2],
    [3.5],
    [7.8]
])

predictions = task.predict(outputs)

print(predictions)