from metrics import classification_metrics


def apply_threshold(
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).float()

    return predictions


def generate_thresholds(
    start=0.1,
    end=0.9,
    step=0.1
):

    thresholds = []

    current = start

    while current <= end:

        thresholds.append(
            round(current, 2)
        )

        current += step

    return thresholds


def threshold_experiment(
    probabilities,
    targets,
    thresholds
):

    results = {}

    for threshold in thresholds:

        predictions = apply_threshold(
            probabilities,
            threshold
        )

        metrics = classification_metrics(
            targets,
            predictions
        )

        results[threshold] = metrics

    return results


def find_threshold_by_metric(
    results,
    metric
):

    if not results:

        raise ValueError(
            "results cannot be empty"
        )

    valid_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1"
    }

    if metric not in valid_metrics:

        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    best_threshold = None
    best_value = -1

    for threshold, metrics in results.items():

        value = metrics[metric]

        if value > best_value:

            best_value = value
            best_threshold = threshold

    return best_threshold, best_value


def find_best_threshold(
    results
):

    return find_threshold_by_metric(
        results,
        "f1"
    )