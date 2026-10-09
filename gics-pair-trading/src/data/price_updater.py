from datetime import timedelta

import pandas as pd

from src.data.price_quality import validate_price_data
from src.data.price_storage import (
    get_price_path,
    load_prices,
    save_prices,
)
from src.data.yahoo_provider import YahooPriceProvider


def update_ticker(
    ticker: str,
    provider=None,
) -> dict:
    """Create or incrementally update one ticker's price history."""

    if provider is None:
        provider = YahooPriceProvider()

    path = get_price_path(ticker)

    # ---------------------------------------------------------
    # First download
    # ---------------------------------------------------------
    if not path.exists():

        df = provider.get_daily_prices(
            ticker=ticker,
            start=pd.Timestamp("2021-01-01").date(),
        )

        quality = validate_price_data(
            df,
            ticker,
        )

        if quality["status"] == "FAILED":
            raise ValueError(
                f"Initial download failed quality check: "
                f"{quality}"
            )

        save_prices(df, ticker)

        return {
            "ticker": ticker,
            "mode": "initial",
            "rows_added": len(df),
            "total_rows": len(df),
            "last_date": df["Date"].max(),
            "quality": quality,
        }

    # ---------------------------------------------------------
    # Incremental update
    # ---------------------------------------------------------
    existing = load_prices(ticker)

    last_date = existing["Date"].max()

    start_date = (
        last_date + timedelta(days=1)
    ).date()

    today = pd.Timestamp.today().date()

    # Already up to date
    if start_date > today:
        return {
            "ticker": ticker,
            "mode": "up_to_date",
            "rows_added": 0,
            "total_rows": len(existing),
            "last_date": last_date,
            "quality": validate_price_data(
                existing,
                ticker,
            ),
        }

    new_data = provider.get_daily_prices(
        ticker=ticker,
        start=start_date,
    )

    if new_data.empty:
        return {
            "ticker": ticker,
            "mode": "no_new_data",
            "rows_added": 0,
            "total_rows": len(existing),
            "last_date": last_date,
            "quality": validate_price_data(
                existing,
                ticker,
            ),
        }

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------
    combined = pd.concat(
        [existing, new_data],
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values("Date")
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Quality check after merge
    # ---------------------------------------------------------
    quality = validate_price_data(
        combined,
        ticker,
    )

    if quality["status"] == "FAILED":
        raise ValueError(
            f"Incremental update failed quality check: "
            f"{quality}"
        )

    save_prices(
        combined,
        ticker,
    )

    return {
        "ticker": ticker,
        "mode": "incremental",
        "rows_added": len(new_data),
        "total_rows": len(combined),
        "last_date": combined["Date"].max(),
        "quality": quality,
    }