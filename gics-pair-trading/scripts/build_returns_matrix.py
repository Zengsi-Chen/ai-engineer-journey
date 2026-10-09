from pathlib import Path

import pandas as pd


PRICE_PATH = Path("data/processed/close_prices.parquet")
OUTPUT_PATH = Path("data/processed/returns.parquet")


def main():
    print("=" * 70)
    print("Building Daily Returns Matrix")
    print("=" * 70)

    prices = pd.read_parquet(PRICE_PATH)

    print(f"Price matrix shape: {prices.shape}")
    print(
        f"Price matrix range: "
        f"{prices.index.min().date()} -> {prices.index.max().date()}"
    )
    print()

    # Calculate returns independently for each ticker.
    # Do NOT drop missing observations globally.
    returns = prices.apply(lambda s: s.dropna().pct_change())

    # The first observation has no previous price.
    # Keeping it as NaN is intentional.
    returns = returns.sort_index()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(OUTPUT_PATH, index=True)

    print("=" * 70)
    print("Returns Matrix Summary")
    print("=" * 70)
    print(f"Shape: {returns.shape}")
    print(
        f"Range: "
        f"{returns.index.min().date()} -> {returns.index.max().date()}"
    )
    print()

    print("Missing observations:")
    missing = returns.isna().sum()
    missing_rate = returns.isna().mean()

    report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_rate": missing_rate,
        }
    ).sort_values("missing_count", ascending=False)

    print(report.to_string())

    print()
    print("Pairwise return observations:")

    pairs = [
        ("AEM", "NEM"),
        ("IVN", "LUN"),
        ("IVN", "FCX"),
        ("GEV", "VST"),
    ]

    for ticker_a, ticker_b in pairs:
        pair_returns = returns[[ticker_a, ticker_b]].dropna()

        print(
            f"  {ticker_a} / {ticker_b}: "
            f"{len(pair_returns)} observations"
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("Returns matrix build: PASSED")


if __name__ == "__main__":
    main()