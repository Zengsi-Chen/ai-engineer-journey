from metrics import (
    ClassificationMetrics,
    RegressionMetrics
)


def create_metrics(task):

    if task == "binary_classification":

        return ClassificationMetrics()

    if task == "multiclass_classification":

        return ClassificationMetrics()

    if task == "regression":

        return RegressionMetrics()

    raise ValueError(
        f"Unknown task: {task}"
    )