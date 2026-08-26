def train_one_epoch(
    model,
    dataloader,
    task,
    optimizer
):

    model.train()

    criterion = task.create_loss()

    total_loss = 0.0

    for inputs, targets in dataloader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(dataloader)
    )

    return average_loss