import torch

def predictions_from_logits(logits, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    return probabilities, predictions