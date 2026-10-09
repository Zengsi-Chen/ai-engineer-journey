from pathlib import Path

import pandas as pd


RAW_PRICE_DIR = Path("data/raw/prices")


def get_price_path(ticker: str) -> Path:
    """Return the raw Parquet path for one ticker."""

    return RAW_PRICE_DIR / f"{ticker}.parquet"


def save_prices(df: pd.DataFrame, ticker: str) -> Path:
    """Save one ticker's daily prices to Parquet."""

    RAW_PRICE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = get_price_path(ticker)

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    ).dt.tz_localize(None)

    df = (
        df.sort_values("Date")
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    df.to_parquet(
        path,
        index=False,
    )

    return path


def load_prices(ticker: str) -> pd.DataFrame:
    """Load one ticker's raw daily prices."""

    path = get_price_path(ticker)

    if not path.exists():
        raise FileNotFoundError(
            f"Price file not found: {path}"
        )

    df = pd.read_parquet(path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    ).dt.tz_localize(None)

    return df