import pytest
import torch

from medseg.models.factory import create_model
from medseg.training.optimizer import create_optimizer


def test_create_adamw():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )


def test_create_adam():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    optimizer = create_optimizer(
        model=model,
        name="adam",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )


def test_optimizer_learning_rate():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=5e-4,
    )

    assert optimizer.param_groups[0]["lr"] == 5e-4


def test_invalid_optimizer():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
    )

    with pytest.raises(ValueError):

        create_optimizer(
            model=model,
            name="invalid_optimizer",
        )


def test_invalid_learning_rate():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
    )

    with pytest.raises(ValueError):

        create_optimizer(
            model=model,
            learning_rate=0.0,
        )


def test_negative_weight_decay():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
    )

    with pytest.raises(ValueError):

        create_optimizer(
            model=model,
            weight_decay=-1e-4,
        )


def test_optimizer_updates_parameters():

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(
        1,
        3,
        64,
        64,
    )

    target = torch.randint(
        0,
        2,
        (1, 1, 64, 64),
    ).float()

    first_parameter = next(
        model.parameters()
    )

    before = first_parameter.detach().clone()

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(
        output,
        target,
    )

    loss.backward()

    optimizer.step()

    after = first_parameter.detach()

    assert not torch.equal(
        before,
        after,
    )