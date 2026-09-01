from model import TransferLearningResNet
from finetuning import (
    FineTuningController,
    get_parameter_stats,
)


def test_feature_extraction_parameter_stats():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    controller.freeze_backbone()

    stats = get_parameter_stats(model)

    assert stats["total"] > 0
    assert stats["trainable"] > 0
    assert stats["frozen"] > 0

    assert (
        stats["trainable"]
        + stats["frozen"]
        == stats["total"]
    )

    assert 0 < stats["trainable_ratio"] < 1

def test_partial_finetuning_parameter_stats():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    controller.unfreeze_layer4()

    stats = get_parameter_stats(model)

    assert stats["trainable"] > 0
    assert stats["frozen"] > 0

    assert (
        stats["trainable"]
        + stats["frozen"]
        == stats["total"]
    )

def test_full_finetuning_parameter_stats():

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    controller.unfreeze_all()

    stats = get_parameter_stats(model)

    assert stats["trainable"] == stats["total"]

    assert stats["frozen"] == 0

    assert stats["trainable_ratio"] == 1.0

