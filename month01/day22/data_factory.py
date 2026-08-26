from dataset import create_dataloaders
from binary_dataset import create_binary_dataloaders


def create_data_loaders(config):

    source = config.source.lower()

    if source == "mnist":

        return create_dataloaders(
            batch_size=config.batch_size
        )

    elif source == "binary":

        return create_binary_dataloaders(
            batch_size=config.batch_size
        )

    else:

        raise ValueError(
            f"Unknown data source: {source}"
        )