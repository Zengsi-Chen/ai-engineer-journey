import torch


class ThresholdOptimizer:

    def __init__(
        self,
        thresholds=None,
        metric="f1"
    ):

        if thresholds is None:

            thresholds = torch.arange(
                0.1,
                1.0,
                0.1
            )

        self.thresholds = thresholds
        self.metric = metric


    def optimize(
        self,
        probabilities,
        targets
    ):

        best_threshold = None
        best_score = None

        results = []


        for threshold in self.thresholds:

            predictions = (
                probabilities
                >= threshold
            ).long()


            score = self._compute_metric(
                predictions,
                targets
            )


            results.append(
                {
                    "threshold": float(threshold),
                    "score": score
                }
            )


            if (
                best_score is None
                or score > best_score
            ):

                best_score = score
                best_threshold = float(
                    threshold
                )


        return {
            "best_threshold": best_threshold,
            "best_score": best_score,
            "results": results
        }


    def _compute_metric(
        self,
        predictions,
        targets
    ):

        if self.metric == "f1":

            tp = (
                (predictions == 1)
                & (targets == 1)
            ).sum().float()

            fp = (
                (predictions == 1)
                & (targets == 0)
            ).sum().float()

            fn = (
                (predictions == 0)
                & (targets == 1)
            ).sum().float()


            precision = (
                tp / (tp + fp)
                if tp + fp > 0
                else torch.tensor(0.0)
            )


            recall = (
                tp / (tp + fn)
                if tp + fn > 0
                else torch.tensor(0.0)
            )


            if precision + recall == 0:

                return 0.0


            return (
                2
                * precision
                * recall
                / (precision + recall)
            ).item()


        raise ValueError(
            f"Unknown metric: {self.metric}"
        )