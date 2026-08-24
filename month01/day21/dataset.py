import torch

from torch.utils.data import Dataset, DataLoader


# ============================================================
# Data Source
# ============================================================

class SyntheticDataSource:

    def __init__(
        self,
        num_samples=100
    ):

        self.num_samples = num_samples


    def load(self):

        x = (
            torch.arange(
                1,
                self.num_samples + 1
            )
            .reshape(-1, 1)
            .float()
            / self.num_samples
        )

        y = (
            x >= 0.5
        ).float()

        return x, y


def create_data_source(
    config
):

    source_name = (
        config.data.source.lower()
    )

    if source_name == "synthetic":

        return SyntheticDataSource()

    raise ValueError(
        f"Unknown data source: {source_name}"
    )


# ============================================================
# Dataset
# ============================================================

class ClassificationDataset(Dataset):

    def __init__(
        self,
        x,
        y
    ):

        self.x = x
        self.y = y


    def __len__(self):

        return len(self.x)


    def __getitem__(
        self,
        index
    ):

        return (
            self.x[index],
            self.y[index]
        )


class RegressionDataset(Dataset):

    def __init__(
        self,
        x,
        y
    ):

        self.x = x
        self.y = y


    def __len__(self):

        return len(self.x)


    def __getitem__(
        self,
        index
    ):

        return (
            self.x[index],
            self.y[index]
        )


def create_dataset(
    x,
    y,
    task_type
):

    task_type = task_type.lower()


    if task_type == "classification":

        return ClassificationDataset(
            x,
            y
        )


    if task_type == "regression":

        return RegressionDataset(
            x,
            y
        )


    raise ValueError(
        f"Unknown task type: {task_type}"
    )


# ============================================================
# Dataset Factory
# ============================================================

def create_datasets(
    x,
    y,
    config
):

    train_ratio = config.data.train_ratio
    val_ratio = config.data.val_ratio
    test_ratio = config.data.test_ratio


    total_ratio = (
        train_ratio
        + val_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-6:

        raise ValueError(
            "train_ratio + val_ratio + "
            "test_ratio must equal 1.0"
        )


    n = len(x)


    train_end = int(
        n * train_ratio
    )


    val_end = (
        train_end
        + int(
            n * val_ratio
        )
    )


    train_dataset = create_dataset(
        x[:train_end],
        y[:train_end],
        config.task.task_type
    )


    val_dataset = create_dataset(
        x[train_end:val_end],
        y[train_end:val_end],
        config.task.task_type
    )


    test_dataset = create_dataset(
        x[val_end:],
        y[val_end:],
        config.task.task_type
    )


    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    config
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# Complete Data Pipeline
# ============================================================

def create_data_pipeline(
    config
):

    data_source = create_data_source(
        config
    )


    x, y = data_source.load()


    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets(
        x,
        y,
        config
    )


    return create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config
    )