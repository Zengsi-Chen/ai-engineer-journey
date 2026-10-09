from abc import ABC, abstractmethod
from datetime import date


class PriceProvider(ABC):
    """Abstract interface for historical market data providers."""

    @abstractmethod
    def get_daily_prices(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
    ):
        """Return daily OHLCV data for one ticker."""
        raise NotImplementedError