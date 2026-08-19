def classification_metrics(y_true, y_pred):

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()

    tn = ((y_true == 0) & (y_pred == 0)).sum().item()

    fp = ((y_true == 0) & (y_pred == 1)).sum().item()

    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }