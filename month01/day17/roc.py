import torch


def roc_curve(
    targets,
    probabilities
):

    targets = targets.view(-1).long()
    probabilities = probabilities.view(-1)

    thresholds = torch.linspace(
        0.0,
        1.0,
        101
    )

    fprs = []
    tprs = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).long()

        tp = (
            ((predictions == 1) & (targets == 1))
            .sum()
            .item()
        )

        fp = (
            ((predictions == 1) & (targets == 0))
            .sum()
            .item()
        )

        tn = (
            ((predictions == 0) & (targets == 0))
            .sum()
            .item()
        )

        fn = (
            ((predictions == 0) & (targets == 1))
            .sum()
            .item()
        )

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    return (
        torch.tensor(fprs),
        torch.tensor(tprs)
    )