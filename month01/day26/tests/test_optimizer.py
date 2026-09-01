from model import TransferLearningResNet
from finetuning import FineTuningController
from optimizer import (
    create_differential_lr_optimizer,
)


def test_differential_learning_rates():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    controller.unfreeze_layer4()

    optimizer = create_differential_lr_optimizer(
        model,
        backbone_lr=1e-5,
        layer4_lr=1e-4,
        classifier_lr=1e-3,
    )

    learning_rates = [
        group["lr"]
        for group in optimizer.param_groups
    ]

    assert 1e-4 in learning_rates
    assert 1e-3 in learning_rates

