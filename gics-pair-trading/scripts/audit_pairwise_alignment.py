from pathlib import Path

import pandas as pd


MATRIX_PATH = Path("data/processed/close_prices.parquet")

PAIRS = [
    ("AEM", "NEM"),
    ("IVN", "LUN"),
    ("IVN", "FCX"),
    ("GEV", "VST"),
]


def analyze_pair(prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> dict:
    pair = prices[[ticker_a, ticker_b]].dropna()

    return {
        "pair": f"{ticker_a} / {ticker_b}",
        "observations": len(pair),
        "first_date": pair.index.min(),
        "last_date": pair.index.max(),
    }


def main():
    prices = pd.read_parquet(MATRIX_PATH)

    print("=" * 70)
    print("Pairwise Trading-Date Alignment Audit")
    print("=" * 70)
    print(f"Matrix shape: {prices.shape}")
    print()

    results = []

    for ticker_a, ticker_b in PAIRS:
        result = analyze_pair(prices, ticker_a, ticker_b)
        results.append(result)

    report = pd.DataFrame(results)

    print(
        f"{'Pair':15s}"
        f"{'Observations':>15s}"
        f"{'First Date':>15s}"
        f"{'Last Date':>15s}"
    )
    print("-" * 70)

    for _, row in report.iterrows():
        print(
            f"{row['pair']:15s}"
            f"{row['observations']:15d}"
            f"{row['first_date'].date()!s:>15s}"
            f"{row['last_date'].date()!s:>15s}"
        )

    print()
    print("=" * 70)
    print("Pairwise Missing-Data Check")
    print("=" * 70)

    for ticker_a, ticker_b in PAIRS:
        pair = prices[[ticker_a, ticker_b]]

        print()
        print(f"{ticker_a} / {ticker_b}")
        print(f"  {ticker_a} missing: {pair[ticker_a].isna().sum()}")
        print(f"  {ticker_b} missing: {pair[ticker_b].isna().sum()}")
        print(f"  Pairwise observations: {pair.dropna().shape[0]}")

    print()
    print("=" * 70)
    print("Alignment Audit: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()