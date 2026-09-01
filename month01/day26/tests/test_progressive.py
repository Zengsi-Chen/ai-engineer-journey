from model import TransferLearningResNet
from finetuning import FineTuningController
from progressive import (
    FineTuningStage,
    apply_strategy,
    ProgressiveFineTuner,
)


def test_stage1_feature_extraction():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    stage = FineTuningStage(
        name="head",
        strategy="feature_extraction",
        epochs=5,
    )

    apply_strategy(
        controller,
        stage.strategy,
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.fc.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in model.backbone.layer4.parameters()
    )

def test_stage2_layer4():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    stage = FineTuningStage(
        name="layer4",
        strategy="partial_finetuning",
        epochs=5,
    )

    apply_strategy(
        controller,
        stage.strategy,
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.layer4.parameters()
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.fc.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter
        in model.backbone.layer3.parameters()
    )

def test_stage3_layer3():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    stage = FineTuningStage(
        name="layer3",
        strategy="layer3_finetuning",
        epochs=5,
    )

    apply_strategy(
        controller,
        stage.strategy,
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.layer3.parameters()
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.layer4.parameters()
    )

def test_stage3_layer3():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    stage = FineTuningStage(
        name="layer3",
        strategy="layer3_finetuning",
        epochs=5,
    )

    apply_strategy(
        controller,
        stage.strategy,
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.layer3.parameters()
    )

    assert all(
        parameter.requires_grad
        for parameter
        in model.backbone.layer4.parameters()
    )