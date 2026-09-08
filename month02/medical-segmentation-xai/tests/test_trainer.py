import torch
from torch.utils.data import DataLoader, TensorDataset

from medseg.models.factory import create_model
from medseg.training.losses import BCEDiceLoss
from medseg.training.optimizer import create_optimizer
from medseg.training.trainer import Trainer


def create_dummy_loader():
    images = torch.rand(4, 3, 64, 64)
    masks = (torch.rand(4, 1, 64, 64) > 0.5).float()

    dataset = TensorDataset(images, masks)

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )


def test_trainer_runs():
    train_loader = create_dummy_loader()
    val_loader = create_dummy_loader()

    model = create_model(
        name="unet",
        features=(8, 16, 32, 64),
    )

    criterion = BCEDiceLoss()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        max_epochs=2,
    )

    result = trainer.fit()

    assert len(result.history) == 2
    assert result.best_epoch is None
    assert result.best_val_dice is None
    assert result.stopped_early is False


def test_trainer_rejects_invalid_epochs():
    train_loader = create_dummy_loader()
    val_loader = create_dummy_loader()

    model = create_model(
        name="unet",
        features=(8, 16, 32, 64),
    )

    criterion = BCEDiceLoss()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
    )

    try:
        Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            max_epochs=0,
        )
        assert False
    except ValueError:
        assert True