import torch


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def evaluate_test(
    model,
    dataloader,
    device,
):
    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.append(
                predictions.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    accuracy = correct / total

    predictions = torch.cat(
        all_predictions
    )

    labels = torch.cat(
        all_labels
    )

    return accuracy, predictions, labels


def confusion_matrix(
    predictions,
    labels,
    num_classes=10,
):
    matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64,
    )

    for true_label, predicted_label in zip(
        labels,
        predictions,
    ):
        matrix[
            true_label,
            predicted_label,
        ] += 1

    return matrix

def per_class_accuracy(
    matrix,
):
    results = {}

    for class_index in range(
        matrix.size(0)
    ):
        total = matrix[
            class_index
        ].sum().item()

        correct = matrix[
            class_index,
            class_index,
        ].item()

        if total == 0:
            accuracy = 0.0
        else:
            accuracy = correct / total

        results[class_index] = accuracy

    return results

def most_confused_pairs(
    matrix,
    top_k=10,
):
    pairs = []

    num_classes = matrix.size(0)

    for true_class in range(
        num_classes
    ):
        for predicted_class in range(
            num_classes
        ):
            if (
                true_class
                != predicted_class
            ):
                count = matrix[
                    true_class,
                    predicted_class,
                ].item()

                pairs.append(
                    (
                        count,
                        true_class,
                        predicted_class,
                    )
                )

    pairs.sort(
        reverse=True
    )

    return pairs[:top_k]
