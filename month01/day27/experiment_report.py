from datetime import datetime
from pathlib import Path


def get_best_experiment(
    results,
):
    """
    Return the experiment with the
    highest best accuracy.
    """

    return max(
        results.items(),
        key=lambda item: (
            item[1]["result"].best_accuracy
        ),
    )


def calculate_tradeoff(
    baseline_result,
    comparison_result,
):
    """
    Calculate differences relative
    to the baseline result.
    """

    accuracy_difference = (
        comparison_result.best_accuracy
        - baseline_result.best_accuracy
    )

    time_difference = (
        comparison_result.training_time
        - baseline_result.training_time
    )

    return (
        accuracy_difference,
        time_difference,
    )


def generate_experiment_report(
    results,
    report_path,
):
    """
    Generate a Markdown experiment report.
    """

    report_path = Path(
        report_path
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = results[
        "uniform_lr"
    ]["result"]

    best_name, best_data = (
        get_best_experiment(
            results
        )
    )

    best_result = best_data[
        "result"
    ]

    lines = []

    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    lines.append(
        "# Day 27 Experiment Report"
    )

    lines.append("")

    lines.append(
        f"Generated: "
        f"{datetime.now().isoformat()}"
    )

    lines.append("")

    # --------------------------------------------------
    # Experiment Setup
    # --------------------------------------------------

    lines.append(
        "## Experiment Setup"
    )

    lines.append("")

    lines.append(
        "The following learning-rate "
        "strategies were compared:"
    )

    lines.append("")

    lines.append(
        "1. Uniform Learning Rate"
    )

    lines.append(
        "2. Manual Discriminative "
        "Learning Rates"
    )

    lines.append(
        "3. Automatic Layer-wise "
        "Learning-rate Decay"
    )

    lines.append("")

    lines.append(
        "All experiments used the same:"
    )

    lines.append("")

    lines.append(
        "- Pretrained checkpoint"
    )

    lines.append(
        "- Target domain"
    )

    lines.append(
        "- Fine-tuning strategy"
    )

    lines.append(
        "- Number of epochs"
    )

    lines.append(
        "- Batch size"
    )

    lines.append(
        "- Weight decay"
    )

    lines.append("")

    # --------------------------------------------------
    # Results Table
    # --------------------------------------------------

    lines.append(
        "## Experiment Results"
    )

    lines.append("")

    lines.append(
        "| Strategy | Best Accuracy | "
        "Best Epoch | Training Time (s) | "
        "Δ Accuracy | Δ Time (s) |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for name, data in results.items():

        result = data[
            "result"
        ]

        (
            accuracy_difference,
            time_difference,
        ) = calculate_tradeoff(
            baseline_result=baseline,
            comparison_result=result,
        )

        lines.append(
            f"| {name} | "
            f"{result.best_accuracy:.4f} | "
            f"{result.best_epoch} | "
            f"{result.training_time:.2f} | "
            f"{accuracy_difference:+.4f} | "
            f"{time_difference:+.2f} |"
        )

    lines.append("")

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    lines.append(
        "## Accuracy Ranking"
    )

    lines.append("")

    ranked_results = sorted(
        results.items(),
        key=lambda item: (
            item[1]["result"].best_accuracy
        ),
        reverse=True,
    )

    for rank, (
        name,
        data,
    ) in enumerate(
        ranked_results,
        start=1,
    ):

        result = data[
            "result"
        ]

        lines.append(
            f"{rank}. **{name}** — "
            f"{result.best_accuracy:.4f}"
        )

    lines.append("")

    # --------------------------------------------------
    # Conclusion
    # --------------------------------------------------

    lines.append(
        "## Conclusion"
    )

    lines.append("")

    lines.append(
        f"The best experiment was "
        f"**{best_name}** "
        f"with a best validation accuracy "
        f"of **{best_result.best_accuracy:.4f}**."
    )

    lines.append("")

    if best_name == "uniform_lr":

        conclusion = (
            "The more complex learning-rate "
            "strategies did not outperform the "
            "uniform learning rate under the "
            "current experimental conditions."
        )

    elif (
        best_name
        == "manual_discriminative_lr"
    ):

        conclusion = (
            "Manually chosen discriminative "
            "learning rates achieved the best "
            "result under the current "
            "experimental conditions."
        )

    else:

        conclusion = (
            "Automatic layer-wise learning-rate "
            "decay achieved the best result "
            "under the current experimental "
            "conditions."
        )

    lines.append(
        conclusion
    )

    lines.append("")

    lines.append(
        "This conclusion is specific to the "
        "current checkpoint, target-domain "
        "shift, training budget, and "
        "hyperparameter configuration."
    )

    lines.append("")

    # --------------------------------------------------
    # Save Report
    # --------------------------------------------------

    report_text = "\n".join(
        lines
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report_text
        )

    print(
        f"\nExperiment report saved to:"
    )

    print(
        report_path
    )