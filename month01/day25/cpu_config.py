import os

import torch


def configure_cpu(
    num_threads=4,
):

    torch.set_num_threads(
        num_threads
    )

    torch.set_num_interop_threads(
        1
    )

    print(
        "\n=== CPU Configuration ==="
    )

    print(
        f"Logical CPUs: "
        f"{os.cpu_count()}"
    )

    print(
        f"PyTorch Threads: "
        f"{torch.get_num_threads()}"
    )

    print(
        f"PyTorch Interop Threads: "
        f"{torch.get_num_interop_threads()}"
    )