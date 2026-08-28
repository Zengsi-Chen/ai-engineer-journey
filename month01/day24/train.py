import torch
from torch import nn


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):
    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += images.size(0)

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy