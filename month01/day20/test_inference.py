import torch
import pytest

from model import BinaryClassifier
from inference import ClassificationInference


class DummyModel(torch.nn.Module):

    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full(
            (x.shape[0], 1),
            self.logit
        )
    

@pytest.fixture
def classifier():

    model = BinaryClassifier()

    model.load_state_dict(
        torch.load(
            "../day18/best_model.pth",
            map_location="cpu"
        )
    )

    best_threshold = 0.5

    return ClassificationInference(
        model=model,
        threshold=best_threshold
    )


@pytest.fixture
def sample_input():
    return torch.tensor([
        [2.5],
        [5.0],
        [7.5],
        [10.0]
    ])


@pytest.mark.parametrize(
    "values",
    [
        [2.5],
        [2.5, 5.0],
        [2.5, 5.0, 7.5, 10.0],
        [-5.0, 0.0, 5.0]
    ]
)


def test_batch_output_size(classifier, values):

    x = torch.tensor(
        values,
        dtype=torch.float32
    ).reshape(-1, 1)

    result = classifier.predict(x)

    assert result.probability.shape[0] == x.shape[0]
    assert result.prediction.shape[0] == x.shape[0]


def test_probability_range(classifier, sample_input):

    result = classifier.predict(sample_input)

    assert torch.all(result.probability >= 0.0)
    assert torch.all(result.probability <= 1.0)


def test_prediction_values(classifier, sample_input):

    result = classifier.predict(sample_input)

    assert torch.all(
        (result.prediction == 0) |
        (result.prediction == 1)
    )


def test_threshold_logic(classifier, sample_input):

    result = classifier.predict(sample_input)

    expected_prediction = (
        result.probability >= classifier.threshold
    ).long()

    assert torch.equal(
        result.prediction,
        expected_prediction
    )


def test_deterministic_inference(classifier, sample_input):

    result1 = classifier.predict(sample_input)
    result2 = classifier.predict(sample_input)

    assert torch.equal(
        result1.prediction,
        result2.prediction
    )

    assert torch.allclose(
        result1.probability,
        result2.probability
    )


def test_invalid_input_shape(classifier):

    invalid_x = torch.tensor([
        [2.5, 3.0],
        [5.0, 6.0]
    ])

    with pytest.raises(ValueError):
        classifier.predict(invalid_x)


def test_threshold_equal_boundary():

    model = DummyModel(logit=0.0)

    classifier = ClassificationInference(
        model=model,
        threshold=0.5
    )

    x = torch.tensor([
        [1.0]
    ])

    result = classifier.predict(x)

    assert result.probability.item() == pytest.approx(0.5)

    assert result.prediction.item() == 1


@pytest.mark.parametrize(
    "probability, threshold, expected",
    [
        (0.2, 0.5, 0),
        (0.49, 0.5, 0),
        (0.5, 0.5, 1),
        (0.51, 0.5, 1),
        (0.8, 0.5, 1),
    ]
)
def test_threshold_boundary(
    probability,
    threshold,
    expected
):

    logit = torch.logit(
        torch.tensor(probability)
    ).item()

    model = DummyModel(logit)

    classifier = ClassificationInference(
        model=model,
        threshold=threshold
    )

    x = torch.tensor([
        [1.0]
    ])

    result = classifier.predict(x)

    assert result.probability.item() == pytest.approx(
        probability,
        abs=1e-6
    )

    assert result.prediction.item() == expected