import pytest

from medseg.training.early_stopping import EarlyStopping


def test_first_value_is_best():
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=3,
    )

    should_stop = early_stopping.step(
        value=0.50,
        epoch=1,
    )

    assert should_stop is False
    assert early_stopping.best_value == 0.50
    assert early_stopping.best_epoch == 1
    assert early_stopping.num_bad_epochs == 0


def test_improvement_resets_patience():
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=3,
    )

    early_stopping.step(0.50, epoch=1)
    early_stopping.step(0.49, epoch=2)
    early_stopping.step(0.48, epoch=3)

    assert early_stopping.num_bad_epochs == 2

    early_stopping.step(0.60, epoch=4)

    assert early_stopping.best_value == 0.60
    assert early_stopping.best_epoch == 4
    assert early_stopping.num_bad_epochs == 0


def test_early_stopping_triggers():
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=2,
    )

    assert early_stopping.step(0.50, epoch=1) is False
    assert early_stopping.step(0.50, epoch=2) is False
    assert early_stopping.step(0.49, epoch=3) is False
    assert early_stopping.step(0.48, epoch=4) is True
    assert early_stopping.step(0.47, epoch=5) is True


def test_min_mode():
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
    )

    assert early_stopping.step(0.50, epoch=1) is False

    early_stopping.step(0.60, epoch=2)
    early_stopping.step(0.70, epoch=3)

    assert early_stopping.best_value == 0.50

    early_stopping.step(0.40, epoch=4)

    assert early_stopping.best_value == 0.40
    assert early_stopping.num_bad_epochs == 0


def test_min_delta():
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=2,
        min_delta=0.01,
    )

    early_stopping.step(0.50, epoch=1)

    # Improvement is only 0.005, so it does not count.
    early_stopping.step(0.505, epoch=2)

    assert early_stopping.best_value == 0.50
    assert early_stopping.num_bad_epochs == 1

    # Improvement is 0.02, so it counts.
    early_stopping.step(0.52, epoch=3)

    assert early_stopping.best_value == 0.52
    assert early_stopping.best_epoch == 3
    assert early_stopping.num_bad_epochs == 0


def test_invalid_mode():
    with pytest.raises(ValueError):
        EarlyStopping(
            mode="invalid",
        )


def test_invalid_patience():
    with pytest.raises(ValueError):
        EarlyStopping(
            patience=-1,
        )


def test_invalid_min_delta():
    with pytest.raises(ValueError):
        EarlyStopping(
            min_delta=-0.1,
        )