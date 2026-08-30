from torchvision import transforms


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


def create_train_transform():

    return transforms.Compose(
        [
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def create_evaluation_transform():

    return transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )