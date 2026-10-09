from datetime import date

import pandas as pd

from src.data.price_updater import update_ticker


class FakeProvider:
    """Fake price provider used to test incremental updates."""

    def __init__(self):
        self.calls = []

    def get_daily_prices(
        self,
        ticker: str,
        start: date,
        end=None,
    ):
        self.calls.append(
            {
                "ticker": ticker,
                "start": start,
                "end": end,
            }
        )

        dates = pd.date_range(
            start=start,
            periods=2,
            freq="D",
        )

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            }
        )


def test_fake_provider_returns_data():
    provider = FakeProvider()

    df = provider.get_daily_prices(
        ticker="TEST",
        start=date(2026, 10, 1),
    )

    assert len(df) == 2

    assert list(df.columns) == [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    assert provider.calls[0]["ticker"] == "TEST"
    assert provider.calls[0]["start"] == date(2026, 10, 1)



def test_update_ticker_when_already_up_to_date(
    monkeypatch,
    tmp_path,
):
    from src.data import price_updater

    today = pd.Timestamp.today().normalize()

    existing = pd.DataFrame(
        {
            "Date": [today],
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1000],
        }
    )

    fake_path = tmp_path / "AEM.parquet"
    fake_path.touch()

    monkeypatch.setattr(
        price_updater,
        "get_price_path",
        lambda ticker: fake_path,
    )

    monkeypatch.setattr(
        price_updater,
        "load_prices",
        lambda ticker: existing.copy(),
    )

    provider = FakeProvider()

    result = update_ticker(
        ticker="AEM",
        provider=provider,
    )

    assert result["ticker"] == "AEM"
    assert result["mode"] == "up_to_date"
    assert result["rows_added"] == 0
    assert result["total_rows"] == 1

    assert provider.calls == []


def test_incremental_update_with_fake_provider(
    monkeypatch,
    tmp_path,
):
    from src.data import price_updater

    existing_dates = pd.date_range(
        start="2026-10-01",
        end="2026-10-03",
        freq="D",
    )

    existing = pd.DataFrame(
        {
            "Date": existing_dates,
            "Open": [90.0, 91.0, 92.0],
            "High": [92.0, 93.0, 94.0],
            "Low": [89.0, 90.0, 91.0],
            "Close": [91.0, 92.0, 93.0],
            "Volume": [900, 950, 1000],
        }
    )

    provider = FakeProvider()

    saved = {}

    def fake_get_price_path(ticker):
        path = tmp_path / "fake_path"
        path.touch()
        return path

    def fake_load_prices(ticker):
        return existing.copy()

    def fake_save_prices(df, ticker):
        saved["df"] = df.copy()
        saved["ticker"] = ticker

    monkeypatch.setattr(
        price_updater,
        "get_price_path",
        fake_get_price_path,
    )

    monkeypatch.setattr(
        price_updater,
        "load_prices",
        fake_load_prices,
    )

    monkeypatch.setattr(
        price_updater,
        "save_prices",
        fake_save_prices,
    )

    result = update_ticker(
        ticker="TEST",
        provider=provider,
    )

    assert result["mode"] == "incremental"

    assert result["rows_added"] == 2

    assert result["total_rows"] == 5

    assert result["last_date"] == pd.Timestamp(
        "2026-10-05"
    )

    assert len(provider.calls) == 1

    assert provider.calls[0]["ticker"] == "TEST"

    assert provider.calls[0]["start"] == date(
        2026,
        10,
        4,
    )

    assert saved["ticker"] == "TEST"

    assert len(saved["df"]) == 5

    assert (
        saved["df"]["Date"]
        .is_unique
    )

    assert (
        saved["df"]["Date"]
        .is_monotonic_increasing
    )