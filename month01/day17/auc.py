import torch


def auc_score(
    fprs,
    tprs
):

    sorted_indices = torch.argsort(
        fprs
    )

    fprs = fprs[sorted_indices]
    tprs = tprs[sorted_indices]

    auc = torch.trapezoid(
        tprs,
        fprs
    )

    return auc.item()