import torch


def confusion_matrix(
    targets,
    predictions
):

    targets = targets.view(-1).long()
    predictions = predictions.view(-1).long()

    tn = (
        ((predictions == 0) & (targets == 0))
        .sum()
        .item()
    )

    fp = (
        ((predictions == 1) & (targets == 0))
        .sum()
        .item()
    )

    fn = (
        ((predictions == 0) & (targets == 1))
        .sum()
        .item()
    )

    tp = (
        ((predictions == 1) & (targets == 1))
        .sum()
        .item()
    )

    return tn, fp, fn, tp