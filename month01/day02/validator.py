def calculate_max(numbers: list[float]) -> float:
    if not numbers:
        raise ValueError("numbers cannot be empty")

    return max(numbers)


def calculate_average(values: list[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")

    return sum(values) / len(values)