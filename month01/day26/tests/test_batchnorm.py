import torch

from model import TransferLearningResNet
from finetuning import freeze_batchnorm


def test_batchnorm_is_frozen():

    model = TransferLearningResNet(
        num_classes=10
    )

    freeze_batchnorm(model)

    batchnorm_layers = [
        module
        for module in model.modules()
        if isinstance(
            module,
            torch.nn.BatchNorm2d
        )
    ]

    assert len(batchnorm_layers) > 0

    for bn in batchnorm_layers:

        assert bn.training is False

        assert all(
            not parameter.requires_grad
            for parameter
            in bn.parameters()
        )


def test_batchnorm_statistics_do_not_change():

    model = TransferLearningResNet(
        num_classes=10
    )

    model.train()

    freeze_batchnorm(model)

    batchnorm = next(
        module
        for module in model.modules()
        if isinstance(
            module,
            torch.nn.BatchNorm2d
        )
    )

    old_mean = (
        batchnorm.running_mean.clone()
    )

    x = torch.randn(
        8,
        3,
        224,
        224,
    )

    with torch.no_grad():
        model(x)

    new_mean = batchnorm.running_mean

    assert torch.equal(
        old_mean,
        new_mean,
    )

def test_batchnorm_updates_in_train_mode():

    model = TransferLearningResNet(
        num_classes=10
    )

    model.train()

    batchnorm = next(
        module
        for module in model.modules()
        if isinstance(
            module,
            torch.nn.BatchNorm2d
        )
    )

    old_mean = (
        batchnorm.running_mean.clone()
    )

    x = torch.randn(
        8,
        3,
        224,
        224,
    )

    with torch.no_grad():
        model(x)

    new_mean = batchnorm.running_mean

    assert not torch.equal(
        old_mean,
        new_mean,
    )