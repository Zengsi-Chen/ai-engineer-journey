from torch import nn
from torchvision import models


def create_resnet18(
    num_classes=10,
    fine_tune=False,
):
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tuning:
    # unfreeze the last ResNet block
    if fine_tune:
        for param in model.layer4.parameters():
            param.requires_grad = True  

    # Replace classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def get_trainable_parameters(model):
    return (
        param
        for param in model.parameters()
        if param.requires_grad
    )