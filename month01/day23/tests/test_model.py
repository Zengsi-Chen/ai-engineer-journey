import torch

from model import CIFAR10CNN


def test_cifar10_cnn_output_shape():
    model = CIFAR10CNN(num_classes=10)

    x = torch.randn(
        8,
        3,
        32,
        32,
    )

    logits = model(x)

    assert logits.shape == (8, 10)


def test_cifar10_cnn_number_of_classes():
    model = CIFAR10CNN(num_classes=10)

    assert model.classifier.out_features == 10