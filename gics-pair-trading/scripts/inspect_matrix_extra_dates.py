from pathlib import Path

import pandas as pd


MATRIX_PATH = Path("data/processed/close_prices.parquet")


def main():
    prices = pd.read_parquet(MATRIX_PATH)

    print("=" * 70)
    print("Inspecting Matrix Extra Dates")
    print("=" * 70)

    # AEM is a long-history reference ticker.
    reference = prices["AEM"]

    extra_dates = prices.index[reference.isna()]

    print(f"Total matrix dates: {len(prices)}")
    print(f"AEM observations: {reference.notna().sum()}")
    print(f"Dates missing for AEM: {len(extra_dates)}")
    print()

    print("Extra dates:")
    print()

    for date in extra_dates:
        row = prices.loc[date]

        available = row.dropna()

        print(
            f"{date.date()} | "
            f"available={len(available):2d} | "
            f"tickers={', '.join(available.index.tolist())}"
        )

    print()
    print("=" * 70)
    print("Weekday distribution")
    print("=" * 70)

    weekday_counts = pd.Series(
        [date.day_name() for date in extra_dates]
    ).value_counts()

    print(weekday_counts.to_string())


if __name__ == "__main__":
    main()