from model import create_resnet18


def test_differential_learning_rates():
    model = create_resnet18(
        num_classes=10,
        fine_tune=True,
    )

    import torch

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.layer4.parameters(),
                "lr": 1e-4,
            },
            {
                "params": model.fc.parameters(),
                "lr": 1e-3,
            },
        ]
    )

    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3