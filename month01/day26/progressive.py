from dataclasses import dataclass


@dataclass
class FineTuningStage:
    name: str
    strategy: str
    epochs: int
    backbone_lr: float = 1e-5
    layer4_lr: float = 1e-4
    classifier_lr: float = 1e-3

stages = [
    FineTuningStage(
        name="head",
        strategy="feature_extraction",
        epochs=5,
        classifier_lr=1e-3,
    ),

    FineTuningStage(
        name="layer4",
        strategy="partial_finetuning",
        epochs=5,
        layer4_lr=1e-4,
        classifier_lr=1e-3,
    ),

    FineTuningStage(
        name="layer3",
        strategy="layer3_finetuning",
        epochs=5,
        backbone_lr=1e-5,
        layer4_lr=1e-4,
        classifier_lr=1e-3,
    ),
]

def apply_strategy(
    controller,
    strategy: str,
):
    if strategy == "feature_extraction":
        controller.freeze_backbone()

    elif strategy == "partial_finetuning":
        controller.unfreeze_layer4()

    elif strategy == "layer3_finetuning":
        controller.unfreeze_layer3()

    elif strategy == "full_finetuning":
        controller.unfreeze_all()

    else:
        raise ValueError(
            f"Unknown fine-tuning strategy: {strategy}"
        )


class ProgressiveFineTuner:

    def __init__(
        self,
        model,
        controller,
        optimizer_factory,
    ):
        self.model = model
        self.controller = controller
        self.optimizer_factory = optimizer_factory


def prepare_stage(self, stage):

    apply_strategy(
        self.controller,
        stage.strategy,
    )

    optimizer = self.optimizer_factory(
        self.model
    )

    return optimizer