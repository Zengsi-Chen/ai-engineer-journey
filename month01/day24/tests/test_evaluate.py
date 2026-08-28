import torch

from evaluate import evaluate


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.size(0)

        outputs = torch.zeros(
            batch_size,
            10,
        )

        outputs[:, 0] = 1.0

        return outputs


def test_evaluate_returns_valid_metrics():
    images = torch.randn(8, 3, 224, 224)
    labels = torch.zeros(8, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(
        images,
        labels,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
    )

    model = DummyModel()

    loss, accuracy = evaluate(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert loss >= 0
    assert accuracy == 1.0