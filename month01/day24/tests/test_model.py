from torch import nn

from model import create_resnet18


def test_classifier_output_classes():
    model = create_resnet18(num_classes=10)

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 10


def test_backbone_is_frozen():
    model = create_resnet18(num_classes=10)

    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            assert param.requires_grad is False


def test_classifier_is_trainable():
    model = create_resnet18(num_classes=10)

    assert model.fc.weight.requires_grad is True
    assert model.fc.bias.requires_grad is True


def test_backbone_parameters_are_frozen():
    model = create_resnet18(num_classes=10)

    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            assert param.requires_grad is False


def test_fine_tuning_unfreezes_layer4():
    model = create_resnet18(
        num_classes=10,
        fine_tune=True,
    )

    for name, param in model.named_parameters():
        if name.startswith("layer4."):
            assert param.requires_grad is True

        elif name.startswith("fc."):
            assert param.requires_grad is True

        else:
            assert param.requires_grad is False