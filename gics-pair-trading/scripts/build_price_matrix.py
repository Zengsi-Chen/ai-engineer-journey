from pathlib import Path

import pandas as pd

from src.gics.universe import flatten_universe, load_universe
from src.data.price_storage import load_prices


CLOSE_OUTPUT_PATH = Path("data/processed/close_prices.parquet")
OPEN_OUTPUT_PATH = Path("data/processed/open_prices.parquet")


def main():
    config = load_universe()
    records = flatten_universe(config["universe"])
    tickers = [record["ticker"] for record in records]

    print("=" * 70)
    print("Building Open / Close Price Matrices")
    print("=" * 70)
    print(f"Tickers: {len(tickers)}")
    print()

    close_series_list = []
    open_series_list = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Loading {ticker}...")

        df = load_prices(ticker)

        close_series = (
            df[["Date", "Close"]]
            .copy()
            .rename(columns={"Close": ticker})
            .set_index("Date")
        )

        open_series = (
            df[["Date", "Open"]]
            .copy()
            .rename(columns={"Open": ticker})
            .set_index("Date")
        )

        close_series_list.append(close_series)
        open_series_list.append(open_series)

    close_prices = pd.concat(
        close_series_list,
        axis=1,
        join="outer",
    ).sort_index()

    open_prices = pd.concat(
        open_series_list,
        axis=1,
        join="outer",
    ).sort_index()

    CLOSE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    close_prices.to_parquet(
        CLOSE_OUTPUT_PATH,
        index=True,
    )

    open_prices.to_parquet(
        OPEN_OUTPUT_PATH,
        index=True,
    )

    print()
    print("=" * 70)
    print("Price Matrix Summary")
    print("=" * 70)

    print(f"Close shape: {close_prices.shape}")
    print(f"Open shape:  {open_prices.shape}")

    print(f"Start: {close_prices.index.min().date()}")
    print(f"End:   {close_prices.index.max().date()}")

    print()
    print("Close missing observations:")

    close_missing = close_prices.isna().sum()
    close_missing_rate = close_prices.isna().mean()

    close_report = pd.DataFrame(
        {
            "missing_count": close_missing,
            "missing_rate": close_missing_rate,
        }
    ).sort_values(
        "missing_count",
        ascending=False,
    )

    print(close_report.to_string())

    print()
    print("Open missing observations:")

    open_missing = open_prices.isna().sum()
    open_missing_rate = open_prices.isna().mean()

    open_report = pd.DataFrame(
        {
            "missing_count": open_missing,
            "missing_rate": open_missing_rate,
        }
    ).sort_values(
        "missing_count",
        ascending=False,
    )

    print(open_report.to_string())

    print()
    print(
        "Close common dates across all tickers:",
        close_prices.dropna().shape[0],
    )

    print(
        "Open common dates across all tickers:",
        open_prices.dropna().shape[0],
    )

    print()
    print(f"Saved Close: {CLOSE_OUTPUT_PATH}")
    print(f"Saved Open:  {OPEN_OUTPUT_PATH}")
    print("Open / Close price matrix build: PASSED")


if __name__ == "__main__":
    main()