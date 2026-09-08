import time
from pathlib import Path

import torch

from medseg.models.factory import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def benchmark_model(
    name: str,
    features: tuple[int, ...],
    image_size: int = 256,
    batch_size: int = 1,
    warmup: int = 2,
    iterations: int = 5,
) -> dict:
    model = create_model(
        name=name,
        in_channels=3,
        out_channels=1,
        features=features,
    )

    model.eval()

    parameters = count_parameters(model)

    x = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
    )

    with torch.no_grad():

        for _ in range(warmup):
            _ = model(x)

        start = time.perf_counter()

        for _ in range(iterations):
            _ = model(x)

        elapsed = time.perf_counter() - start

    average_time = elapsed / iterations
    throughput = batch_size / average_time

    return {
        "name": name,
        "features": list(features),
        "parameters": parameters,
        "parameter_memory_mb": (
            parameters * 4 / 1024 / 1024
        ),
        "batch_size": batch_size,
        "image_size": image_size,
        "average_forward_seconds": average_time,
        "throughput_images_per_second": throughput,
    }


def main() -> None:
    results = []

    configurations = [
        (
            "unet_baseline",
            (32, 64, 128, 256),
        ),
        (
            "unet_cpu",
            (16, 32, 64, 128),
        ),
    ]

    for name, features in configurations:

        result = benchmark_model(
            name="unet",
            features=features,
        )

        result["configuration"] = name

        results.append(result)

    print("=" * 70)
    print("Day 34 Final Model Benchmark")
    print("=" * 70)

    for result in results:

        print(
            f"\nConfiguration: "
            f"{result['configuration']}"
        )

        print(
            f"Features: "
            f"{result['features']}"
        )

        print(
            f"Parameters: "
            f"{result['parameters']:,}"
        )

        print(
            f"Parameter Memory: "
            f"{result['parameter_memory_mb']:.2f} MB"
        )

        print(
            f"Average Forward: "
            f"{result['average_forward_seconds']:.4f} s"
        )

        print(
            f"Throughput: "
            f"{result['throughput_images_per_second']:.2f} "
            f"images/sec"
        )

    baseline = results[0]
    cpu_model = results[1]

    parameter_reduction = (
        1
        - cpu_model["parameters"]
        / baseline["parameters"]
    )

    throughput_improvement = (
        cpu_model["throughput_images_per_second"]
        / baseline["throughput_images_per_second"]
    )

    print("\n" + "=" * 70)
    print("Architecture Trade-off")
    print("=" * 70)

    print(
        f"Parameter reduction: "
        f"{parameter_reduction * 100:.2f}%"
    )

    print(
        f"Throughput improvement: "
        f"{throughput_improvement:.2f}x"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()