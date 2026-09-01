import torch


def trainable_parameters(module):
    return [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def create_differential_lr_optimizer(
    model,
    backbone_lr=1e-5,
    layer4_lr=1e-4,
    classifier_lr=1e-3,
):
    parameter_groups = []

    for module, lr in [
        (model.backbone.layer1, backbone_lr),
        (model.backbone.layer2, backbone_lr),
        (model.backbone.layer3, backbone_lr),
        (model.backbone.layer4, layer4_lr),
        (model.backbone.fc, classifier_lr),
    ]:

        parameters = trainable_parameters(module)

        if parameters:
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                }
            )

    return torch.optim.AdamW(
        parameter_groups
    )