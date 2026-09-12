import numpy as np

from medseg.data.dataloader import create_segmentation_dataloader


MANIFEST_PATH = "artifacts/splits/split_manifest.csv"


def test_train_dataset_has_no_augmentation_when_disabled():
    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        augmentation="none",
    )

    assert loader.dataset.transform is None


def test_train_dataset_has_augmentation_when_enabled():
    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        augmentation="hflip",
    )

    assert loader.dataset.transform is not None


def test_validation_dataset_has_no_augmentation():
    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=256,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    assert loader.dataset.transform is None


def test_test_dataset_has_no_augmentation():
    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="test",
        image_size=256,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    assert loader.dataset.transform is None


def test_validation_sample_is_deterministic():
    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=256,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    image_1, mask_1 = loader.dataset[0]
    image_2, mask_2 = loader.dataset[0]

    assert np.array_equal(
        image_1.numpy(),
        image_2.numpy(),
    )

    assert np.array_equal(
        mask_1.numpy(),
        mask_2.numpy(),
    )


def test_test_sample_is_deterministic():
    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="test",
        image_size=256,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    image_1, mask_1 = loader.dataset[0]
    image_2, mask_2 = loader.dataset[0]

    assert np.array_equal(
        image_1.numpy(),
        image_2.numpy(),
    )

    assert np.array_equal(
        mask_1.numpy(),
        mask_2.numpy(),
    )