from model import TransferLearningResNet
from finetuning import (
    FineTuningController,
    get_parameter_stats,
)


def analyze_strategy(strategy):

    model = TransferLearningResNet(
        num_classes=10
    )

    controller = FineTuningController(model)

    if strategy == "feature_extraction":
        controller.freeze_backbone()

    elif strategy == "partial_finetuning":
        controller.unfreeze_layer4()

    elif strategy == "full_finetuning":
        controller.unfreeze_all()

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    return get_parameter_stats(model)


def compare_strategies():

    strategies = [
        "feature_extraction",
        "partial_finetuning",
        "full_finetuning",
    ]

    for strategy in strategies:

        stats = analyze_strategy(strategy)

        print(f"\nStrategy: {strategy}")
        print(
            f"Total:     {stats['total']:,}"
        )
        print(
            f"Trainable: {stats['trainable']:,}"
        )
        print(
            f"Frozen:    {stats['frozen']:,}"
        )
        print(
            f"Ratio:     {stats['trainable_ratio']:.2%}"
        )


if __name__ == "__main__":
    compare_strategies()