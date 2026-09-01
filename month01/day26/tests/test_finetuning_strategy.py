from model import ResNet18

from finetuning_strategy import (
    create_fine_tuning_strategies,
    apply_fine_tuning_strategy,
    get_trainable_parameter_count,
    get_total_parameter_count,
)

def test_three_strategies():

    strategies = (
        create_fine_tuning_strategies()
    )

    names = [
        strategy.name
        for strategy in strategies
    ]

    assert names == [
        "feature_extraction",
        "partial_fine_tuning",
        "full_fine_tuning",
    ]

def test_feature_extraction():

    model = ResNet18(
        num_classes=10
    )

    strategy = (
        create_fine_tuning_strategies()[0]
    )

    apply_fine_tuning_strategy(
        model,
        strategy,
    )

    trainable = (
        get_trainable_parameter_count(
            model
        )
    )

    classifier_params = sum(
        parameter.numel()
        for parameter in model.classifier.parameters()
    )

    assert trainable == classifier_params


def test_full_fine_tuning():

    model = ResNet18(
        num_classes=10
    )

    strategy = (
        create_fine_tuning_strategies()[2]
    )

    apply_fine_tuning_strategy(
        model,
        strategy,
    )

    assert (
        get_trainable_parameter_count(model)
        ==
        get_total_parameter_count(model)
    )

def test_partial_fine_tuning():

    model = ResNet18(
        num_classes=10
    )

    strategy = (
        create_fine_tuning_strategies()[1]
    )

    apply_fine_tuning_strategy(
        model,
        strategy,
    )

    for name, parameter in model.named_parameters():

        if (
            name.startswith("layer3")
            or name.startswith("layer4")
            or name.startswith("classifier")
        ):
            assert parameter.requires_grad

        else:
            assert not parameter.requires_grad