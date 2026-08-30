import torch
import torch.nn as nn

from resnet import (
    ResidualBlock,
    DeepResidualCNN,
    ProjectionResidualBlock,
    BasicBlock,
    Bottleneck,
)
from resnet18 import ResNet18


def test_residual_block_output_shape():

    block = ResidualBlock(
        channels=8
    )

    x = torch.randn(
        4,
        8,
        8,
        8,
    )

    output = block(x)

    assert output.shape == x.shape


def test_deep_residual_cnn_output_shape():

    model = DeepResidualCNN(
        num_blocks=4
    )

    x = torch.randn(
        4,
        1,
        8,
        8,
    )

    output = model(x)

    assert output.shape == (
        4,
        2,
    )


def test_residual_block_gradient():

    block = ResidualBlock(
        channels=8
    )

    x = torch.randn(
        4,
        8,
        8,
        8,
        requires_grad=True,
    )

    output = block(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None


def test_identity_shortcut():

    block = ProjectionResidualBlock(
        in_channels=8,
        out_channels=8,
        stride=1,
    )

    assert isinstance(
        block.shortcut,
        nn.Identity,
    )

def test_projection_shortcut():

    block = ProjectionResidualBlock(
        in_channels=8,
        out_channels=16,
        stride=2,
    )

    assert isinstance(
        block.shortcut,
        nn.Conv2d,
    )

def test_projection_block_output_shape():

    block = ProjectionResidualBlock(
        in_channels=8,
        out_channels=16,
        stride=2,
    )

    x = torch.randn(
        4,
        8,
        8,
        8,
    )

    output = block(x)

    assert output.shape == (
        4,
        16,
        4,
        4,
    )


def test_basic_block_identity_output_shape():

    block = BasicBlock(
        in_channels=8,
        out_channels=8,
        stride=1,
    )

    x = torch.randn(
        4,
        8,
        8,
        8,
    )

    output = block(x)

    assert output.shape == (
        4,
        8,
        8,
        8,
    )


def test_basic_block_projection_output_shape():

    block = BasicBlock(
        in_channels=8,
        out_channels=16,
        stride=2,
    )

    x = torch.randn(
        4,
        8,
        8,
        8,
    )

    output = block(x)

    assert output.shape == (
        4,
        16,
        4,
        4,
    )


def test_basic_block_uses_batchnorm():

    block = BasicBlock(
        in_channels=8,
        out_channels=8,
    )

    assert isinstance(
        block.bn1,
        nn.BatchNorm2d,
    )

    assert isinstance(
        block.bn2,
        nn.BatchNorm2d,
    )


def test_bottleneck_output_shape():

    block = Bottleneck(
        in_channels=256,
        base_channels=64,
        stride=1,
    )

    x = torch.randn(
        4,
        256,
        8,
        8,
    )

    output = block(x)

    assert output.shape == (
        4,
        256,
        8,
        8,
    )


def test_bottleneck_downsample():

    block = Bottleneck(
        in_channels=256,
        base_channels=128,
        stride=2,
    )

    x = torch.randn(
        4,
        256,
        8,
        8,
    )

    output = block(x)

    assert output.shape == (
        4,
        512,
        4,
        4,
    )


def test_bottleneck_expansion():

    assert Bottleneck.expansion == 4


def test_resnet18_output_shape():

    model = ResNet18(
        num_classes=10
    )

    x = torch.randn(
        4,
        3,
        32,
        32,
    )

    output = model(x)

    assert output.shape == (
        4,
        10,
    )

def test_resnet18_stage_channels():

    model = ResNet18()

    x = torch.randn(
        2,
        3,
        32,
        32,
    )

    x = model.stem(x)

    x = model.layer1(x)

    assert x.shape == (
        2,
        64,
        32,
        32,
    )

    x = model.layer2(x)

    assert x.shape == (
        2,
        128,
        16,
        16,
    )

    x = model.layer3(x)

    assert x.shape == (
        2,
        256,
        8,
        8,
    )

    x = model.layer4(x)

    assert x.shape == (
        2,
        512,
        4,
        4,
    )