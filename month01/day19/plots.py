import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    metrics,
    save_path=None
):
    """
    Plot binary classification confusion matrix.
    """

    tp = metrics["tp"]
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]

    matrix = [
        [tn, fp],
        [fn, tp]
    ]

    fig, ax = plt.subplots()

    ax.imshow(matrix)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center"
            )

    ax.set_title("Confusion Matrix")

    fig.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


def calculate_roc_curve(
    targets,
    probabilities,
    thresholds=None
):
    """
    Calculate FPR and TPR for different thresholds.
    """

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)

    if thresholds is None:
        thresholds = np.linspace(
            0.0,
            1.0,
            101
        )

    fpr = []
    tpr = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tp = np.sum(
            (targets == 1) &
            (predictions == 1)
        )

        tn = np.sum(
            (targets == 0) &
            (predictions == 0)
        )

        fp = np.sum(
            (targets == 0) &
            (predictions == 1)
        )

        fn = np.sum(
            (targets == 1) &
            (predictions == 0)
        )

        false_positive_rate = (
            fp / (fp + tn)
            if fp + tn > 0
            else 0.0
        )

        true_positive_rate = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        fpr.append(false_positive_rate)
        tpr.append(true_positive_rate)

    return (
        np.array(fpr),
        np.array(tpr)
    )

def plot_roc_curve(
    targets,
    probabilities,
    save_path=None
):
    """
    Plot ROC curve.
    """

    fpr, tpr = calculate_roc_curve(
        targets,
        probabilities
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label="ROC"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title("ROC Curve")

    ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)