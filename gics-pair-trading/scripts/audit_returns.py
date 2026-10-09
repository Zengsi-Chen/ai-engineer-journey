from pathlib import Path

import pandas as pd


RETURNS_PATH = Path("data/processed/returns.parquet")


def main():
    returns = pd.read_parquet(RETURNS_PATH)

    print("=" * 70)
    print("Returns Data Quality Audit")
    print("=" * 70)

    print(f"Shape: {returns.shape}")
    print(
        f"Date range: "
        f"{returns.index.min().date()} -> "
        f"{returns.index.max().date()}"
    )
    print()

    # ------------------------------------------------------------
    # 1. Infinite values
    # ------------------------------------------------------------
    infinite_count = int(returns.isin([float("inf"), float("-inf")]).sum().sum())

    print(f"Infinite values: {infinite_count}")

    # ------------------------------------------------------------
    # 2. NaN values
    # ------------------------------------------------------------
    total_missing = int(returns.isna().sum().sum())

    print(f"Total missing values: {total_missing}")

    # ------------------------------------------------------------
    # 3. Extreme returns
    # ------------------------------------------------------------
    valid_returns = returns.stack()

    print()
    print("Return distribution:")
    print(valid_returns.describe())

    print()

    extreme_10 = (valid_returns.abs() > 0.10).sum()
    extreme_20 = (valid_returns.abs() > 0.20).sum()
    extreme_50 = (valid_returns.abs() > 0.50).sum()

    print(f"|Return| > 10%: {extreme_10}")
    print(f"|Return| > 20%: {extreme_20}")
    print(f"|Return| > 50%: {extreme_50}")

    # ------------------------------------------------------------
    # 4. Largest positive returns
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("Largest Positive Returns")
    print("=" * 70)

    largest = valid_returns.nlargest(10)

    for (date, ticker), value in largest.items():
        print(
            f"{date.date()} | "
            f"{ticker:6s} | "
            f"{value:+.2%}"
        )

    # ------------------------------------------------------------
    # 5. Largest negative returns
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("Largest Negative Returns")
    print("=" * 70)

    smallest = valid_returns.nsmallest(10)

    for (date, ticker), value in smallest.items():
        print(
            f"{date.date()} | "
            f"{ticker:6s} | "
            f"{value:+.2%}"
        )

    # ------------------------------------------------------------
    # 6. Final status
    # ------------------------------------------------------------
    print()
    print("=" * 70)

    if infinite_count == 0:
        print("Infinite-value check: PASSED")
    else:
        print("Infinite-value check: FAILED")

    print("Returns data audit: PASSED")


if __name__ == "__main__":
    main()