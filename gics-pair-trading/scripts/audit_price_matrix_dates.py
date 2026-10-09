from pathlib import Path

import pandas as pd


MATRIX_PATH = Path("data/processed/close_prices.parquet")


def main():
    prices = pd.read_parquet(MATRIX_PATH)

    print("=" * 70)
    print("Price Matrix Date Coverage Audit")
    print("=" * 70)

    print(f"Matrix shape: {prices.shape}")
    print(f"Matrix start: {prices.index.min().date()}")
    print(f"Matrix end:   {prices.index.max().date()}")
    print()

    records = []

    for ticker in prices.columns:
        series = prices[ticker]

        valid = series.dropna()

        records.append(
            {
                "ticker": ticker,
                "observations": len(valid),
                "first_date": valid.index.min(),
                "last_date": valid.index.max(),
                "missing": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
            }
        )

    report = pd.DataFrame(records)

    report = report.sort_values(
        ["missing", "ticker"],
        ascending=[False, True],
    )

    print("Date coverage by ticker:")
    print()

    for _, row in report.iterrows():
        print(
            f"{row['ticker']:6s} "
            f"obs={row['observations']:4d} "
            f"missing={row['missing']:4d} "
            f"missing_rate={row['missing_rate']:.2%} "
            f"{row['first_date'].date()} -> "
            f"{row['last_date'].date()}"
        )

    print()
    print("=" * 70)
    print("Missing-count distribution")
    print("=" * 70)

    print(
        report["missing"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 70)
    print("Tickers with exactly 30 missing observations")
    print("=" * 70)

    exactly_30 = report[report["missing"] == 30]

    print(
        exactly_30[
            [
                "ticker",
                "observations",
                "first_date",
                "last_date",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()