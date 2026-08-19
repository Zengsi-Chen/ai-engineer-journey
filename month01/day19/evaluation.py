import torch

from metrics import calculate_metrics


def evaluate_model(
    model,
    dataloader,
    threshold=0.5
):
    """
    Evaluate a binary classification model.
    """

    model.eval()

    all_probabilities = []
    all_targets = []

    with torch.no_grad():

        for x_batch, y_batch in dataloader:

            logits = model(x_batch)

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities
            )

            all_targets.append(
                y_batch
            )

    probabilities = torch.cat(
        all_probabilities
    ).flatten()

    targets = torch.cat(
        all_targets
    ).flatten()

    predictions = (
        probabilities >= threshold
    ).float()

    metrics = calculate_metrics(
        targets,
        predictions
    )

    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "targets": targets,
        "predictions": predictions
    }

def evaluate_thresholds(
    targets,
    probabilities,
    thresholds
):
    """
    Evaluate classification metrics
    across multiple thresholds.
    """

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).float()

        metrics = calculate_metrics(
            targets,
            predictions
        )

        results.append({
            "threshold": float(threshold),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    return results


def select_best_threshold(
    threshold_results,
    min_recall=0.95
):
    """
    Select the best threshold based on F1.

    Two thresholds are selected:

    1. Best F1 overall
    2. Best F1 with Recall >= min_recall
    """

    best_f1 = max(
        threshold_results,
        key=lambda x: x["f1"]
    )

    valid_results = [
        result
        for result in threshold_results
        if result["recall"] >= min_recall
    ]

    if valid_results:
        best_recall_constrained = max(
            valid_results,
            key=lambda x: x["f1"]
        )
    else:
        best_recall_constrained = None

    return {
        "best_f1": best_f1,
        "best_recall_constrained": (
            best_recall_constrained
        )
    }

def evaluate_predictions(
    targets,
    probabilities,
    threshold
):
    """
    Evaluate predictions using
    pre-computed probabilities.
    """

    predictions = (
        probabilities >= threshold
    ).float()

    metrics = calculate_metrics(
        targets,
        predictions
    )

    return {
        "threshold": float(threshold),
        "metrics": metrics
    }

def run_evaluation(
    model,
    val_loader,
    threshold,
    thresholds,
    min_recall
):
    """
    Run the complete evaluation pipeline.

    Steps:
    1. Model inference
    2. Normal evaluation
    3. Threshold sweep
    4. Best threshold selection
    5. Final evaluation
    """

    # -------------------------
    # 1. Model inference
    # -------------------------

    results = evaluate_model(
        model,
        val_loader,
        threshold=threshold
    )


    # -------------------------
    # 2. Threshold analysis
    # -------------------------

    threshold_results = evaluate_thresholds(
        results["targets"],
        results["probabilities"],
        thresholds
    )


    # -------------------------
    # 3. Select best threshold
    # -------------------------

    best_thresholds = select_best_threshold(
        threshold_results,
        min_recall=min_recall
    )


    # -------------------------
    # 4. Final evaluation
    # -------------------------

    selected_threshold = (
        best_thresholds["best_f1"]["threshold"]
    )

    final_results = evaluate_predictions(
        results["targets"],
        results["probabilities"],
        selected_threshold
    )


    return {
        "targets": results["targets"],
        "probabilities": results["probabilities"],
        "metrics": results["metrics"],
        "threshold_results": threshold_results,
        "best_thresholds": best_thresholds,
        "final_results": final_results
    }


def print_evaluation_summary(
    evaluation_results,
    min_recall
):
    """
    Print a human-readable evaluation summary.
    """

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

    final_metrics = final_results["metrics"]

    # =========================
    # Validation Metrics
    # =========================

    print()
    print("Validation Metrics")
    print("------------------")

    print(
        f"Accuracy:  {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"F1:        {metrics['f1']:.4f}"
    )

    # =========================
    # Threshold Analysis
    # =========================

    print()
    print("Threshold Analysis")
    print("------------------")

    for row in threshold_results:

        print(
            f"Threshold: {row['threshold']:.1f} "
            f"Accuracy: {row['accuracy']:.4f} "
            f"Precision: {row['precision']:.4f} "
            f"Recall: {row['recall']:.4f} "
            f"F1: {row['f1']:.4f}"
        )

    # =========================
    # Best Threshold
    # =========================

    print()
    print("Best Threshold Selection")
    print("------------------------")

    best_f1 = best_thresholds["best_f1"]

    print(
        f"Best F1 Threshold: "
        f"{best_f1['threshold']:.1f}"
    )

    print(
        f"Best F1: "
        f"{best_f1['f1']:.4f}"
    )

    constrained = (
        best_thresholds[
            "best_recall_constrained"
        ]
    )

    if constrained is not None:

        print(
            f"Best Threshold "
            f"(Recall >= {min_recall:.2f}): "
            f"{constrained['threshold']:.1f}"
        )

        print(
            f"Constrained F1: "
            f"{constrained['f1']:.4f}"
        )

    # =========================
    # Final Evaluation
    # =========================

    print()
    print("Final Evaluation")
    print("----------------")

    print(
        f"Selected Threshold: "
        f"{final_results['threshold']:.1f}"
    )

    print(
        f"Accuracy:  {final_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {final_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {final_metrics['recall']:.4f}"
    )

    print(
        f"F1:        {final_metrics['f1']:.4f}"
    )