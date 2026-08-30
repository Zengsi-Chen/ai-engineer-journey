import torch

from cifar10_augmentation import (
    create_train_transform,
    create_evaluation_transform,
)


def test_train_transform_output_shape():

    transform = (
        create_train_transform()
    )

    image = torch.randint(
        0,
        256,
        (
            3,
            32,
            32,
        ),
        dtype=torch.uint8,
    )

    from torchvision.transforms.functional import (
        to_pil_image,
    )

    image = to_pil_image(image)

    output = transform(image)

    assert output.shape == (
        3,
        32,
        32,
    )


def test_evaluation_transform_output_shape():

    transform = (
        create_evaluation_transform()
    )

    image = torch.randint(
        0,
        256,
        (
            3,
            32,
            32,
        ),
        dtype=torch.uint8,
    )

    from torchvision.transforms.functional import (
        to_pil_image,
    )

    image = to_pil_image(image)

    output = transform(image)

    assert output.shape == (
        3,
        32,
        32,
    )