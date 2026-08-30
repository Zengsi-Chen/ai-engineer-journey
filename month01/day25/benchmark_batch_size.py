import time

import torch

from cifar10_data import (
    create_cifar10_loaders,
)

from resnet18 import ResNet18


NUM_THREADS = 4


def benchmark(
    batch_size,
):

    torch.set_num_threads(
        NUM_THREADS
    )

    torch.manual_seed(42)

    train_loader, _, _ = (
        create_cifar10_loaders(
            batch_size=batch_size,
            train_samples=1000,
            validation_samples=100,
            seed=42,
            num_workers=0,
        )
    )

    model = ResNet18()

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    model.train()

    start_time = (
        time.perf_counter()
    )

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    samples_per_second = (
        len(train_loader.dataset)
        / elapsed_time
    )

    return (
        elapsed_time,
        samples_per_second,
    )


def main():

    print(
        f"PyTorch Threads: "
        f"{NUM_THREADS}"
    )

    for batch_size in [
        32,
        64,
        128,
    ]:

        elapsed_time, throughput = (
            benchmark(
                batch_size
            )
        )

        print(
            f"\nBatch Size: "
            f"{batch_size}"
        )

        print(
            f"Time: "
            f"{elapsed_time:.2f} seconds"
        )

        print(
            f"Throughput: "
            f"{throughput:.2f} "
            f"samples/sec"
        )


if __name__ == "__main__":
    main()