from pathlib import Path

import pandas as pd

from src.gics.universe import flatten_universe, load_universe
from src.data.price_storage import load_prices


def main():
    config = load_universe()
    records = flatten_universe(config["universe"])
    tickers = [record["ticker"] for record in records]

    price_dir = Path("data/raw/prices")

    print("=" * 70)
    print("Local Parquet Price Integrity Audit")
    print("=" * 70)
    print(f"Expected tickers: {len(tickers)}")
    print(f"Price directory: {price_dir}")
    print()

    passed = []
    warnings = []
    failed = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Checking {ticker}...", end=" ")

        try:
            path = price_dir / f"{ticker}.parquet"

            if not path.exists():
                failed.append(
                    {
                        "ticker": ticker,
                        "errors": [f"Missing file: {path}"],
                        "warnings": [],
                    }
                )
                print("FAIL")
                continue

            df = load_prices(ticker)

            errors = []
            ticker_warnings = []

            required_columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:
                errors.append(
                    f"Missing columns: {missing_columns}"
                )

            if df.empty:
                errors.append("DataFrame is empty")

            if not df.empty:
                duplicate_dates = int(
                    df["Date"].duplicated().sum()
                )

                if duplicate_dates > 0:
                    errors.append(
                        f"Duplicate dates: {duplicate_dates}"
                    )

                if not df["Date"].is_monotonic_increasing:
                    errors.append(
                        "Dates are not sorted ascending"
                    )

                missing_values = (
                    df[required_columns]
                    .isna()
                    .sum()
                    .sum()
                )

                if missing_values > 0:
                    errors.append(
                        f"Missing values: {int(missing_values)}"
                    )

                observations = len(df)

                if observations < 400:
                    ticker_warnings.append(
                        f"Only {observations} observations"
                    )

                start_date = df["Date"].min()
                end_date = df["Date"].max()

                print(
                    f"{'FAIL' if errors else 'PASS'} "
                    f"({observations} rows, "
                    f"{start_date.date()} -> "
                    f"{end_date.date()})"
                )
            else:
                print("FAIL")

            if errors:
                failed.append(
                    {
                        "ticker": ticker,
                        "errors": errors,
                        "warnings": ticker_warnings,
                    }
                )
            elif ticker_warnings:
                warnings.append(
                    {
                        "ticker": ticker,
                        "errors": [],
                        "warnings": ticker_warnings,
                    }
                )
            else:
                passed.append(ticker)

        except Exception as exc:
            failed.append(
                {
                    "ticker": ticker,
                    "errors": [str(exc)],
                    "warnings": [],
                }
            )
            print("FAIL")

    print()
    print("=" * 70)
    print("Local Parquet Audit Summary")
    print("=" * 70)
    print(f"Passed: {len(passed)}")
    print(f"Passed with warning: {len(warnings)}")
    print(f"Failed: {len(failed)}")
    print(f"Total: {len(tickers)}")

    if warnings:
        print()
        print("Tickers with warnings:")
        for result in warnings:
            print(
                f"  {result['ticker']}: "
                f"{result['warnings']}"
            )

    if failed:
        print()
        print("Failed tickers:")
        for result in failed:
            print(
                f"  {result['ticker']}: "
                f"{result['errors']}"
            )

        raise SystemExit(1)

    print()
    print("Local Parquet audit: PASSED")


if __name__ == "__main__":
    main()