import torch

class Metric:

    def __init__(
        self,
        name
    ):

        self.name = name

    def compute(
        self,
        predictions,
        targets
    ):

        raise NotImplementedError


class AccuracyMetric(Metric):

    def __init__(self):

        super().__init__("accuracy")


    def compute(
        self,
        predictions,
        targets
    ):

        predictions = predictions.long()
        targets = targets.long()

        correct = (
            predictions == targets
        ).sum()

        return (
            correct.float()
            / targets.numel()
        ).item()   


class PrecisionMetric(Metric):

    def __init__(self):

        super().__init__("precision")


    def compute(
        self,
        predictions,
        targets
    ):

        predictions = predictions.long()
        targets = targets.long()

        tp = (
            (predictions == 1)
            & (targets == 1)
        ).sum().float()

        fp = (
            (predictions == 1)
            & (targets == 0)
        ).sum().float()

        if tp + fp == 0:

            return 0.0

        return (
            tp / (tp + fp)
        ).item()

class RecallMetric(Metric):

    def __init__(self):

        super().__init__("recall")


    def compute(
        self,
        predictions,
        targets
    ):

        predictions = predictions.long()
        targets = targets.long()

        tp = (
            (predictions == 1)
            & (targets == 1)
        ).sum().float()

        fn = (
            (predictions == 0)
            & (targets == 1)
        ).sum().float()

        if tp + fn == 0:

            return 0.0

        return (
            tp / (tp + fn)
        ).item()

class F1Metric(Metric):

    def __init__(self):

        super().__init__("f1")

        self.precision = PrecisionMetric()
        self.recall = RecallMetric()


    def compute(
        self,
        predictions,
        targets
    ):

        precision = self.precision.compute(
            predictions,
            targets
        )

        recall = self.recall.compute(
            predictions,
            targets
        )

        if precision + recall == 0:

            return 0.0

        return (
            2
            * precision
            * recall
            / (precision + recall)
        )

class MAEMetric(Metric):

    def __init__(self):

        super().__init__("mae")


    def compute(
        self,
        predictions,
        targets
    ):

        return torch.mean(
            torch.abs(
                predictions - targets
            )
        ).item()


class MSEMetric(Metric):

    def __init__(self):

        super().__init__("mse")


    def compute(
        self,
        predictions,
        targets
    ):

        return torch.mean(
            (predictions - targets) ** 2
        ).item()


class RMSEMetric(Metric):

    def __init__(self):

        super().__init__("rmse")


    def compute(
        self,
        predictions,
        targets
    ):

        mse = torch.mean(
            (predictions - targets) ** 2
        )

        return torch.sqrt(
            mse
        ).item()


def create_metrics(
    config
):

    task_type = (
        config.task.task_type.lower()
    )


    if task_type == "classification":

        return [
            AccuracyMetric(),
            PrecisionMetric(),
            RecallMetric(),
            F1Metric()
        ]


    if task_type == "regression":

        return [
            MAEMetric(),
            MSEMetric(),
            RMSEMetric()
        ]


    raise ValueError(
        f"Unknown task type: {task_type}"
    )