from dataclasses import dataclass

import torch


@dataclass
class ClassificationResult:
    probability: torch.Tensor
    prediction: torch.Tensor


def preprocess(x):

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(
            x,
            dtype=torch.float32
        )

    x = x.float()

    if x.ndim == 1:
        x = x.unsqueeze(1)

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D input [batch_size, 1], "
            f"but got shape {tuple(x.shape)}"
        )

    if x.shape[1] != 1:
        raise ValueError(
            f"Expected 1 feature, "
            f"but got {x.shape[1]}"
        )

    return x


class ClassificationInference:

    def __init__(
        self,
        model,
        threshold=0.5,
        device="cpu"
    ):

        self.model = model
        self.threshold = threshold
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def predict(self, x):

        x = preprocess(x)
        x = x.to(self.device)

        with torch.no_grad():

            output = self.model(x)

            probability = torch.sigmoid(output)

            prediction = (
                probability >= self.threshold
            ).long()

        return ClassificationResult(
            probability=probability,
            prediction=prediction
        )