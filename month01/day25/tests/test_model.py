import torch

from model import (
    ShallowCNN,
    DeepPlainCNN,
    VeryDeepPlainCNN,
)


def test_shallow_cnn_output_shape():

    model = ShallowCNN()

    x = torch.randn(4, 1, 8, 8)

    output = model(x)

    assert output.shape == (4, 2)


def test_deep_plain_cnn_output_shape():

    model = DeepPlainCNN()

    x = torch.randn(4, 1, 8, 8)

    output = model(x)

    assert output.shape == (4, 2)


def test_shallow_cnn_gradient():

    model = ShallowCNN()

    x = torch.randn(4, 1, 8, 8)

    y = torch.randint(0, 2, (4,))

    criterion = torch.nn.CrossEntropyLoss()

    output = model(x)

    loss = criterion(output, y)

    loss.backward()

    first_parameter = next(model.parameters())

    assert first_parameter.grad is not None


def test_deep_plain_cnn_gradient():

    model = DeepPlainCNN()

    x = torch.randn(4, 1, 8, 8)

    y = torch.randint(0, 2, (4,))

    criterion = torch.nn.CrossEntropyLoss()

    output = model(x)

    loss = criterion(output, y)

    loss.backward()

    first_parameter = next(model.parameters())

    assert first_parameter.grad is not None


def test_very_deep_cnn_output_shape():

    model = VeryDeepPlainCNN(
        num_layers=20
    )

    x = torch.randn(4, 1, 8, 8)

    output = model(x)

    assert output.shape == (4, 2)


def test_very_deep_cnn_gradient_exists():

    model = VeryDeepPlainCNN(
        num_layers=20
    )

    x = torch.randn(4, 1, 8, 8)

    y = torch.randint(0, 2, (4,))

    criterion = torch.nn.CrossEntropyLoss()

    output = model(x)

    loss = criterion(output, y)

    loss.backward()

    first_parameter = next(model.parameters())

    assert first_parameter.grad is not None