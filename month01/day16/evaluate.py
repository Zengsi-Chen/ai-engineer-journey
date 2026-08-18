import torch

from metrics import mae, mse, rmse, r2_score


def evaluate(model, x, y):
    model.eval()

    with torch.no_grad():
        y_pred = model(x)

        results = {
            "mae": mae(y_pred, y).item(),
            "mse": mse(y_pred, y).item(),
            "rmse": rmse(y_pred, y).item(),
            "r2": r2_score(y_pred, y).item(),
        }

    return results, y_pred