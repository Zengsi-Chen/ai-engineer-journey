import torch


def evaluate(
    model,
    dataloader,
    task,
    metrics
):

    model.eval()

    criterion = task.create_loss()

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for inputs, targets in dataloader:

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            predictions = task.predict(
                outputs
            )

            total_loss += loss.item()

            all_predictions.append(
                predictions
            )

            all_targets.append(
                targets
            )

    predictions = torch.cat(
        all_predictions,
        dim=0
    )

    targets = torch.cat(
        all_targets,
        dim=0
    )

    metric_results = metrics.compute(
        predictions,
        targets
    )

    average_loss = (
        total_loss / len(dataloader)
    )

    return {
        "loss": average_loss,
        **metric_results
    }