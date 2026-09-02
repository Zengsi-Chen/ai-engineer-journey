from dataclasses import dataclass


@dataclass
class FineTuningStrategy:
    name: str
    freeze_backbone: bool = False
    trainable_layers: tuple = ()

def create_fine_tuning_strategies():

    return [
        FineTuningStrategy(
            name="feature_extraction",
            freeze_backbone=True,
        ),

        FineTuningStrategy(
            name="partial_fine_tuning",
            trainable_layers=(
                "layer3",
                "layer4",
                "classifier",
            ),
        ),

        FineTuningStrategy(
            name="full_fine_tuning",
            freeze_backbone=False,
        ),
    ]


def apply_fine_tuning_strategy(
    model,
    strategy,
):
    for parameter in model.parameters():
        parameter.requires_grad = True

    if strategy.name == "feature_extraction":

        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    elif strategy.name == "partial_fine_tuning":

        for parameter in model.parameters():
            parameter.requires_grad = False

        for layer_name in strategy.trainable_layers:

            layer = getattr(
                model,
                layer_name,
            )

            for parameter in layer.parameters():
                parameter.requires_grad = True

    elif strategy.name == "full_fine_tuning":

        for parameter in model.parameters():
            parameter.requires_grad = True

    else:

        raise ValueError(
            f"Unknown strategy: "
            f"{strategy.name}"
        )

    return model

def get_trainable_parameter_count(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

def get_total_parameter_count(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )