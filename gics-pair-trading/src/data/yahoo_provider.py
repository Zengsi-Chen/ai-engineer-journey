from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from src.data.price_provider import PriceProvider


TICKER_MAP_PATH = Path("config/ticker_map.yaml")


def load_ticker_map() -> dict[str, str]:
    if not TICKER_MAP_PATH.exists():
        return {}

    with TICKER_MAP_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config.get("ticker_map", {})


class YahooPriceProvider(PriceProvider):

    def get_daily_prices(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:

        ticker_map = load_ticker_map()
        yahoo_ticker = ticker_map.get(ticker, ticker)

        end_value = end.isoformat() if end else None

        df = yf.download(
            yahoo_ticker,
            start=start.isoformat(),
            end=end_value,
            interval="1d",
            auto_adjust=False,
            progress=False,
            actions=False,
        )

        if df.empty:
            raise ValueError(
                f"No price data returned for ticker: "
                f"{ticker} (Yahoo ticker: {yahoo_ticker})"
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns for {ticker} "
                f"(Yahoo ticker: {yahoo_ticker}): {missing}"
            )

        df = df[required_columns].copy()

        df.columns.name = None

        df["Date"] = (
            pd.to_datetime(df["Date"])
            .dt.tz_localize(None)
        )

        df = (
            df.sort_values("Date")
            .drop_duplicates(subset="Date")
            .reset_index(drop=True)
        )

        return df