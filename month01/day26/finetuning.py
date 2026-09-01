import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters inside a module."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters inside a module."""
    for parameter in module.parameters():
        parameter.requires_grad = True


class FineTuningController:

    def __init__(self, model: nn.Module):
        self.model = model

    def freeze_backbone(self):
        freeze_module(self.model.backbone)
        unfreeze_module(self.model.backbone.fc)

    def unfreeze_layer4(self):
        self.freeze_backbone()
        unfreeze_module(self.model.backbone.layer4)

    def unfreeze_layer3(self):
        self.freeze_backbone()
        unfreeze_module(self.model.backbone.layer3)
        unfreeze_module(self.model.backbone.layer4)

    def unfreeze_all(self):
        unfreeze_module(self.model.backbone)

def get_parameter_stats(model: nn.Module):
    total = 0
    trainable = 0

    for parameter in model.parameters():
        num_parameters = parameter.numel()

        total += num_parameters

        if parameter.requires_grad:
            trainable += num_parameters

    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": trainable / total,
    }

def print_parameter_stats(model: nn.Module):
    stats = get_parameter_stats(model)

    print(f"Total parameters:     {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Frozen parameters:    {stats['frozen']:,}")
    print(
        f"Trainable ratio:      "
        f"{stats['trainable_ratio']:.2%}"
    )

def freeze_batchnorm(module: nn.Module) -> None:

    for child in module.modules():

        if isinstance(
            child,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ),
        ):
            child.eval()

            for parameter in child.parameters():
                parameter.requires_grad = False