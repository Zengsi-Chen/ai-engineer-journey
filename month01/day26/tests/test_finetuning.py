from model import TransferLearningResNet
from finetuning import FineTuningController


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_freeze_backbone():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    controller.freeze_backbone()

    assert all(
        not parameter.requires_grad
        for name, parameter
        in model.backbone.named_parameters()
        if name != "fc.weight"
        and name != "fc.bias"
    )

    assert all(
        parameter.requires_grad
        for parameter in model.backbone.fc.parameters()
    )

def test_unfreeze_layer4():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    controller.unfreeze_layer4()

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.layer4.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in model.backbone.layer3.parameters()
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.fc.parameters()
    )

def test_unfreeze_all():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    controller.unfreeze_all()

    assert all(
        parameter.requires_grad
        for parameter in model.parameters()
    )