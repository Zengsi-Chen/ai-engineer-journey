import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy,
    )


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy,
    )