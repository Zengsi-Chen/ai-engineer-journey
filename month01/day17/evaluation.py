import torch
import matplotlib.pyplot as plt

from metrics import classification_metrics
from confusion_matrix import confusion_matrix
from roc import roc_curve
from auc import auc_score


def plot_roc_curve(
    fprs,
    tprs,
    auc,
    save_path=None
):

    plt.plot(
        fprs.numpy(),
        tprs.numpy(),
        label=f"ROC AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

    plt.show()

    # plt.close()



def evaluate_classification(
    model,
    val_loader,
    thresholds=None,
    plot_roc=False,
    roc_save_path=None
):

    # -------------------------
    # 1. Default thresholds
    # -------------------------

    if thresholds is None:
        thresholds = [0.5]

    # -------------------------
    # 2. Evaluation mode
    # -------------------------

    model.eval()

    all_logits = []
    all_targets = []

    # -------------------------
    # 3. Collect logits
    # -------------------------

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            logits = model(batch_x)

            all_logits.append(logits)
            all_targets.append(batch_y)

    all_logits = torch.cat(
        all_logits
    )

    all_targets = torch.cat(
        all_targets
    )

    # -------------------------
    # 4. Logits -> Probability
    # -------------------------

    probabilities = torch.sigmoid(
        all_logits
    )

    # -------------------------
    # 5. Threshold experiments
    # -------------------------

    threshold_results = {}

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).float()

        metrics = classification_metrics(
            all_targets,
            predictions
        )

        cm = confusion_matrix(
            all_targets,
            predictions
        )

        threshold_results[threshold] = {
            "confusion_matrix": cm,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        }

    # -------------------------
    # 6. ROC / AUC
    # -------------------------

    fprs, tprs = roc_curve(
        all_targets,
        probabilities
    )

    auc = auc_score(
        fprs,
        tprs
    )

    # -------------------------
    # 7. Plot ROC
    # -------------------------
    if plot_roc:

        plot_roc_curve(
            fprs,
            tprs,
            auc,
            save_path=roc_save_path
        )

    # -------------------------
    # 8. Return results
    # -------------------------

    return {
        "probabilities": probabilities,
        "targets": all_targets,
        "thresholds": threshold_results,
        "fprs": fprs,
        "tprs": tprs,
        "auc": auc
    }
    


    