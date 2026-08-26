import torch
from torch import nn

from model import CIFAR10CNN


def test_single_training_step():
    model = CIFAR10CNN(num_classes=10)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    images = torch.randn(
        8,
        3,
        32,
        32,
    )

    labels = torch.randint(
        0,
        10,
        (8,),
    )

    optimizer.zero_grad()

    logits = model(images)

    loss = criterion(
        logits,
        labels,
    )

    loss.backward()

    optimizer.step()

    assert loss.ndim == 0
    assert torch.isfinite(loss)