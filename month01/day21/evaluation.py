import torch
from trainer import move_to_device


class Evaluator:

    def __init__(
        self,
        model,
        metrics,
        output_adapter,
        device="cpu"
    ):

        self.model = model
        self.metrics = metrics
        self.output_adapter = output_adapter
        self.device = device

        self.model.to(
            self.device
        )


    def evaluate(
        self,
        data_loader
    ):

        self.model.eval()

        all_predictions = []
        all_targets = []


        with torch.no_grad():

            for inputs, targets in data_loader:

                inputs = move_to_device(
                    inputs,
                    self.device
                )

                targets = move_to_device(
                    targets,
                    self.device
                )


                output = self.model(
                    inputs
                )


                processed = (
                    self.output_adapter.process(
                        output
                    )
                )


                all_predictions.append(
                        processed["prediction"].cpu()
                    )

                all_targets.append(
                        targets.cpu()
                    )



            predictions = torch.cat(
                all_predictions
            )

            targets = torch.cat(
                all_targets
            )


            results = {}

            for metric in self.metrics:

                results[metric.name] = (
                    metric.compute(
                        predictions,
                        targets
                    )
                )


            return results

        
    def predict(
        self,
        data_loader
    ):

        self.model.eval()

        all_probabilities = []
        all_targets = []


        with torch.no_grad():

            for inputs, targets in data_loader:

                inputs = inputs.to(
                    self.device
                )

                output = self.model(
                    inputs
                )


                processed = (
                    self.output_adapter.process(
                        output
                    )
                )


                if "probability" not in processed:

                    raise ValueError(
                        "Output adapter does not "
                        "provide probability."
                    )


                all_probabilities.append(
                    processed[
                        "probability"
                    ].cpu()
                )

                all_targets.append(
                    targets.cpu()
                )


        probabilities = torch.cat(
            all_probabilities
        )

        targets = torch.cat(
            all_targets
        )


        return probabilities, targets