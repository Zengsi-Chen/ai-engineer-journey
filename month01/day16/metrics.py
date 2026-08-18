import torch


def mae(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def rmse(y_pred, y_true):
    return torch.sqrt(mse(y_pred, y_true))


def r2_score(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot