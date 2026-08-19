import csv
import json
import os

from plots import (
    plot_confusion_matrix,
    plot_roc_curve
)


def save_results(
    evaluation_results,
    metrics_path,
    final_metrics_path,
    threshold_metrics_path,
    best_threshold_path,
    confusion_matrix_path,
    roc_curve_path
):
    """
    Save all evaluation artifacts.
    """

    # =========================
    # 1. Extract results
    # =========================

    metrics = evaluation_results["metrics"]

    threshold_results = (
        evaluation_results["threshold_results"]
    )

    best_thresholds = (
        evaluation_results["best_thresholds"]
    )

    final_results = (
        evaluation_results["final_results"]
    )

    targets = evaluation_results["targets"]

    probabilities = (
        evaluation_results["probabilities"]
    )


    # =========================
    # 2. Save metrics.json
    # =========================

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # =========================
    # 3. Save final_metrics.json
    # =========================

    with open(
        final_metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=4
        )


    # =========================
    # 4. Save threshold CSV
    # =========================

    with open(
        threshold_metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]
        )

        writer.writeheader()

        writer.writerows(
            threshold_results
        )


    # =========================
    # 5. Save best threshold
    # =========================

    with open(
        best_threshold_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_thresholds,
            f,
            indent=4
        )


    # =========================
    # 6. Save confusion matrix
    # =========================

    plot_confusion_matrix(
        metrics,
        save_path=confusion_matrix_path
    )


    # =========================
    # 7. Save ROC curve
    # =========================

    plot_roc_curve(
        targets,
        probabilities,
        save_path=roc_curve_path
    )