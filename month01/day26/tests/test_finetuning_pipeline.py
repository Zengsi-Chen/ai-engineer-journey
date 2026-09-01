import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    )

from finetuning_pipeline import (
    FineTuningResult,
    train_one_epoch,
    evaluate,
    run_fine_tuning,
    )

from finetuning_strategy import (
    create_fine_tuning_strategies,
    )

from model import ResNet18

def create_test_dataloader(
    num_samples=8,
    batch_size=4,
    ):
    torch.manual_seed(42)


    inputs = torch.randn(
        num_samples,
        3,
        32,
        32,
    )

    targets = torch.randint(
        low=0,
        high=10,
        size=(num_samples,),
    )

    dataset = TensorDataset(
        inputs,
        targets,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
    )


def test_train_one_epoch():


    model = ResNet18(
        num_classes=10,
    )

    dataloader = create_test_dataloader()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.CrossEntropyLoss()

    loss, accuracy = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    assert isinstance(
        loss,
        float,
    )

    assert isinstance(
        accuracy,
        float,
    )

    assert loss >= 0.0

    assert (
        0.0
        <= accuracy
        <= 1.0
    )


def test_evaluate():

    model = ResNet18(
        num_classes=10,
    )

    dataloader = create_test_dataloader()

    criterion = torch.nn.CrossEntropyLoss()

    loss, accuracy = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device="cpu",
    )

    assert isinstance(
        loss,
        float,
    )

    assert isinstance(
        accuracy,
        float,
    )

    assert loss >= 0.0

    assert (
        0.0
        <= accuracy
        <= 1.0
    )

    assert model.training is False


def test_run_fine_tuning():


    model = ResNet18(
        num_classes=10,
    )

    train_loader = (
        create_test_dataloader()
    )

    val_loader = (
        create_test_dataloader()
    )

    strategy = (
        create_fine_tuning_strategies()[0]
    )

    result = run_fine_tuning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        strategy=strategy,
        device="cpu",
        epochs=1,
        learning_rate=1e-3,
    )

    assert isinstance(
        result,
        FineTuningResult,
    )

    assert (
        result.strategy_name
        == "feature_extraction"
    )

    assert result.best_epoch == 1

    assert (
        0.0
        <= result.best_accuracy
        <= 1.0
    )

    assert (
        result.final_accuracy
        == result.best_accuracy
    )

