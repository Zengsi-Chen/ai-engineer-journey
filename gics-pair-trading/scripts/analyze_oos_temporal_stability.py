
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts" / "reports" / "oos_bidirectional_equity.csv"
OUT = ROOT / "artifacts" / "reports"

ROLLING_WINDOW = 60
TRADING_DAYS_PER_YEAR = 252


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of a compounded return path."""
    equity = (1.0 + returns).cumprod()
    equity = pd.concat(
        [pd.Series([1.0]), equity.reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def summarize_period(group: pd.DataFrame) -> pd.Series:
    """Summarize one year or quarter using daily portfolio returns."""
    returns = group["daily_return"].astype(float)
    returns = returns.dropna()

    if returns.empty:
        return pd.Series(dtype=float)

    total_return = (1.0 + returns).prod() - 1.0
    daily_std = returns.std(ddof=1)

    sharpe = (
        returns.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if pd.notna(daily_std) and daily_std > 0
        else np.nan
    )

    return pd.Series({
        "observations": len(returns),
        "start_date": group["date"].min().strftime("%Y-%m-%d"),
        "end_date": group["date"].max().strftime("%Y-%m-%d"),
        "total_return_pct": total_return * 100,
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": calculate_max_drawdown(returns) * 100,
        "positive_day_pct": (returns > 0).mean() * 100,
        "mean_active_trades": group["active_trades"].mean(),
        "max_active_trades": group["active_trades"].max(),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT, parse_dates=["date"])
    required = {
        "date", "equity", "daily_return",
        "daily_pnl", "active_trades",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        duplicates = df.loc[df["date"].duplicated(), "date"]
        raise ValueError(f"Duplicate dates found:\n{duplicates}")

    if df["daily_return"].isna().any():
        raise ValueError("daily_return contains missing values.")

    if not np.isfinite(df["daily_return"]).all():
        raise ValueError("daily_return contains non-finite values.")

    print("=" * 72)
    print("OOS TEMPORAL STABILITY ANALYSIS")
    print("=" * 72)
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")
    print(f"First recorded equity: ${df['equity'].iloc[0]:,.2f}")
    print(f"Last recorded equity:  ${df['equity'].iloc[-1]:,.2f}")
    print(f"Compounded daily return: "
          f"{((1 + df['daily_return']).prod() - 1) * 100:.3f}%")

    # Check gaps in the recorded dates. This does not assume
    # that every calendar day should have a row.
    date_gaps = df["date"].diff().dt.days
    large_gaps = df.loc[date_gaps > 4, ["date"]].copy()
    large_gaps["gap_days"] = date_gaps[date_gaps > 4].values

    print(f"\nDate gaps greater than 4 calendar days: {len(large_gaps)}")
    if not large_gaps.empty:
        print(large_gaps.to_string(index=False))

    # Yearly summaries
    df["year"] = df["date"].dt.year
    yearly = (
        df.groupby("year", sort=True)
        .apply(summarize_period, include_groups=False)
        .reset_index()
    )
    yearly.to_csv(OUT / "oos_temporal_yearly.csv", index=False)

    # Quarterly summaries
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    quarterly = (
        df.groupby("quarter", sort=True)
        .apply(summarize_period, include_groups=False)
        .reset_index()
    )
    quarterly.to_csv(OUT / "oos_temporal_quarterly.csv", index=False)

    # Rolling 60-row metrics.
    # These are 60 recorded observations, not necessarily
    # exactly 60 exchange sessions if the source has missing dates.
    r = df["daily_return"].astype(float)
    df["rolling_60_return_pct"] = (
        (1.0 + r).rolling(ROLLING_WINDOW).apply(np.prod, raw=True) - 1.0
    ) * 100

    rolling_mean = r.rolling(ROLLING_WINDOW).mean()
    rolling_std = r.rolling(ROLLING_WINDOW).std(ddof=1)
    df["rolling_60_sharpe"] = (
        rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).where(rolling_std > 0)

    # Rebuild a compounded equity curve from daily returns,
    # normalized to 100 at the start.
    df["rebased_equity"] = (1.0 + r).cumprod() * 100
    df["running_peak"] = df["rebased_equity"].cummax()
    df["drawdown_pct"] = (
        df["rebased_equity"] / df["running_peak"] - 1.0
    ) * 100

    rolling_cols = [
        "date", "daily_return", "equity", "active_trades",
        "rebased_equity", "drawdown_pct",
        "rolling_60_return_pct", "rolling_60_sharpe",
    ]
    df[rolling_cols].to_csv(
        OUT / "oos_temporal_rolling.csv", index=False
    )

    print("\nYEARLY SUMMARY")
    print(yearly.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nQUARTERLY SUMMARY")
    print(quarterly.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot 1: normalized equity and drawdown
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["date"], df["rebased_equity"])
    axes[0].set_title("OOS Compounded Equity (Start = 100)")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["date"], df["drawdown_pct"])
    axes[1].set_title("OOS Drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "oos_temporal_equity_drawdown.png", dpi=150)
    plt.close(fig)

    # Plot 2: rolling 60-observation return and Sharpe
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["date"], df["rolling_60_return_pct"])
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("Rolling 60-Observation Compounded Return")
    axes[0].set_ylabel("Return (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["date"], df["rolling_60_sharpe"])
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Rolling 60-Observation Annualized Sharpe")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "oos_temporal_stability.png", dpi=150)
    plt.close(fig)

    print("\nSaved reports:")
    print(OUT / "oos_temporal_yearly.csv")
    print(OUT / "oos_temporal_quarterly.csv")
    print(OUT / "oos_temporal_rolling.csv")
    print(OUT / "oos_temporal_equity_drawdown.png")
    print(OUT / "oos_temporal_stability.png")


if __name__ == "__main__":
    main()