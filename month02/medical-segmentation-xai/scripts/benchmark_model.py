import time

import torch

from medseg.models.factory import create_model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def benchmark_forward(
    model: torch.nn.Module,
    batch_size: int = 2,
    image_size: int = 256,
    warmup: int = 3,
    iterations: int = 10,
) -> float:
    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    x = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    with torch.no_grad():

        for _ in range(warmup):
            _ = model(x)

        start = time.perf_counter()

        for _ in range(iterations):
            _ = model(x)

        end = time.perf_counter()

    total_time = end - start

    return total_time / iterations


def benchmark_model(
    model_name: str,
    features: tuple[int, ...],
) -> dict:

    model = create_model(
        name=model_name,
        in_channels=3,
        out_channels=1,
        features=features,
    )

    total_params = count_parameters(model)

    trainable_params = count_trainable_parameters(model)

    average_time = benchmark_forward(
        model=model,
        batch_size=2,
        image_size=256,
        warmup=3,
        iterations=10,
    )

    throughput = 2 / average_time

    model_size_mb = (
        total_params * 4
    ) / (1024 ** 2)

    return {
        "name": model_name,
        "features": features,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "average_time": average_time,
        "throughput": throughput,
    }


def print_result(result: dict) -> None:

    print("-" * 50)

    print(f"Model: {result['name']}")

    print(f"Features: {result['features']}")

    print(f"Total parameters: {result['total_params']:,}")

    print(
        f"Trainable parameters: "
        f"{result['trainable_params']:,}"
    )

    print(
        f"Approx. parameter memory: "
        f"{result['model_size_mb']:.2f} MB"
    )

    print(
        f"Average forward time: "
        f"{result['average_time']:.4f} sec"
    )

    print(
        f"Throughput: "
        f"{result['throughput']:.2f} images/sec"
    )


def main() -> None:

    print("=" * 50)

    print("U-Net CPU Architecture Comparison")

    print("=" * 50)

    baseline = benchmark_model(
        model_name="unet",
        features=(32, 64, 128, 256),
    )

    small = benchmark_model(
        model_name="unet",
        features=(16, 32, 64, 128),
    )

    print_result(baseline)

    print_result(small)

    print("=" * 50)


if __name__ == "__main__":
    main()