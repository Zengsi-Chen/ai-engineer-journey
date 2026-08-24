import torch

class OutputAdapter:

    def process(
        self,
        output
    ):

        raise NotImplementedError


class BinaryClassificationAdapter(
    OutputAdapter
):

    def __init__(
        self,
        config
    ):

        self.threshold = config.task.threshold


    def process(
        self,
        output
    ):

        probability = torch.sigmoid(
            output
        )

        prediction = (
            probability
            >= self.threshold
        ).long()


        return {
            "logits": output,
            "probability": probability,
            "prediction": prediction
        }


class RegressionAdapter(
    OutputAdapter
):

    def process(
        self,
        output
    ):

        return {
            "prediction": output
        }


def create_output_adapter(
    config
):

    task_type = (
        config.task.task_type.lower()
    )


    if task_type == "classification":

        return BinaryClassificationAdapter(
            config
        )


    if task_type == "regression":

        return RegressionAdapter(config)


    raise ValueError(
        f"Unknown task type: {task_type}"
    )


