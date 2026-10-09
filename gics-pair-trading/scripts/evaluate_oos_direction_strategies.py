
from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

EVENT_FILE = Path(
    "data/processed/pair_event_backtest_hedged_tier1.parquet"
)
PRICE_DIR = Path("data/raw")
OUTPUT_DIR = Path("artifacts/reports")

THRESHOLD = 2.0
HORIZON = 40

INITIAL_CAPITAL = 100_000.0

# Maximum 20 simultaneously open pair trades.
# Each new trade receives 1/20 of current equity as gross notional.
MAX_OPEN_TRADES = 20
ALLOCATION_PER_TRADE = 1.0 / MAX_OPEN_TRADES

# Sensitivity assumptions, not verified broker quotes.
TRANSACTION_COST_BPS = 5.0
ANNUAL_SHORT_BORROW_RATE = 0.03

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# Load prices
# ============================================================

def load_price_file(ticker, candidates):
    exact = [p for p in candidates if p.stem.upper() == ticker]
    matches = exact or [
        p for p in candidates
        if p.stem.upper().startswith(ticker + "_")
    ]

    if not matches:
        raise FileNotFoundError(
            f"No parquet price file found for {ticker} "
            f"under {PRICE_DIR}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple price files found for {ticker}: {matches}"
        )

    df = pd.read_parquet(matches[0])

    colmap = {str(c).lower(): c for c in df.columns}

    date_col = next(
        (colmap[c] for c in ("date", "datetime", "timestamp")
         if c in colmap),
        None,
    )
    open_col = colmap.get("open")
    close_col = colmap.get("close")

    if not date_col or not open_col or not close_col:
        raise ValueError(
            f"{matches[0]} must contain Date, Open, and Close columns."
        )

    df = df[[date_col, open_col, close_col]].copy()
    df.columns = ["Date", "Open", "Close"]
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    for c in ("Open", "Close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "Close"])
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    df = df.drop_duplicates("Date", keep="last")
    df = df.set_index("Date").sort_index()

    return df


def load_all_prices(tickers):
    files = list(PRICE_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {PRICE_DIR}"
        )

    prices = {}
    for ticker in sorted(tickers):
        prices[ticker] = load_price_file(ticker, files)

    return prices


# ============================================================
# Reconstruct one trade from Open to Close
# ============================================================

def reconstruct_trade(row, prices):
    ticker_a = row["ticker_a"]
    ticker_b = row["ticker_b"]
    signal_type = row["signal_type"]
    beta = float(row["hedge_ratio"])

    entry_date = pd.Timestamp(row["entry_date"]).normalize()
    exit_date = pd.Timestamp(row["exit_date"]).normalize()

    if not np.isfinite(beta) or beta <= 0:
        return None

    if signal_type not in ("LONG_A", "LONG_B"):
        return None

    pa = prices[ticker_a]
    pb = prices[ticker_b]

    # Use dates on which both legs have prices.
    joined = pd.concat(
        [
            pa[["Open", "Close"]].add_prefix("A_"),
            pb[["Open", "Close"]].add_prefix("B_"),
        ],
        axis=1,
        join="inner",
    )

    joined = joined.loc[entry_date:exit_date]

    # Do not silently shift execution to another date.
    if (
        joined.empty
        or entry_date not in joined.index
        or exit_date not in joined.index
    ):
        return None

    joined = joined.loc[entry_date:exit_date]

    if (joined[["A_Open", "A_Close", "B_Open", "B_Close"]] <= 0).any().any():
        return None

    # Initial gross notional is $1:
    # long notional = 1/(1+beta)
    # short notional = beta/(1+beta)
    long_weight = 1.0 / (1.0 + beta)
    short_weight = beta / (1.0 + beta)

    first = joined.iloc[0]

    if signal_type == "LONG_A":
        # Long A, short B.
        shares_a = long_weight / first["A_Open"]
        shares_b = -short_weight / first["B_Open"]
        short_leg = "B"
    else:
        # Long B, short A.
        shares_a = -short_weight / first["A_Open"]
        shares_b = long_weight / first["B_Open"]
        short_leg = "A"

    fee_rate = TRANSACTION_COST_BPS / 10_000.0
    daily_pnl = {}
    previous_close_a = None
    previous_close_b = None

    for date, r in joined.iterrows():
        if date == entry_date:
            # Entry occurs at Open; mark the position to Close.
            delta_a = r["A_Close"] - r["A_Open"]
            delta_b = r["B_Close"] - r["B_Open"]

            # Entry turnover is $1 gross notional.
            pnl = shares_a * delta_a + shares_b * delta_b
            pnl -= fee_rate
        else:
            delta_a = r["A_Close"] - previous_close_a
            delta_b = r["B_Close"] - previous_close_b
            pnl = shares_a * delta_a + shares_b * delta_b

        # Borrow cost based on the current short-leg market value.
        if short_leg == "A":
            short_market_value = abs(shares_a * r["A_Close"])
        else:
            short_market_value = abs(shares_b * r["B_Close"])

        pnl -= (
            short_market_value
            * ANNUAL_SHORT_BORROW_RATE
            / TRADING_DAYS_PER_YEAR
        )

        # Exit occurs at Close. Charge costs on exit turnover.
        if date == exit_date:
            exit_turnover = (
                abs(shares_a * r["A_Close"])
                + abs(shares_b * r["B_Close"])
            )
            pnl -= fee_rate * exit_turnover

        daily_pnl[date] = float(pnl)

        previous_close_a = r["A_Close"]
        previous_close_b = r["B_Close"]

    result = row.to_dict()
    result["_daily_pnl_per_dollar"] = daily_pnl
    result["_trade_return"] = sum(daily_pnl.values())
    result["_entry_date"] = entry_date
    result["_exit_date"] = exit_date
    result["_pair"] = (ticker_a, ticker_b)

    return result


# ============================================================
# Portfolio simulation
# ============================================================

def simulate_portfolio(trades, calendar, strategy_name):
    candidates = sorted(
        [dict(t) for t in trades],
        key=lambda t: (
            t["_entry_date"],
            pd.Timestamp(t["signal_date"]),
            t["ticker_a"],
            t["ticker_b"],
        ),
    )

    entries = {}
    for trade in candidates:
        entries.setdefault(trade["_entry_date"], []).append(trade)

    active = []
    executed = []
    skipped_capacity = 0
    skipped_duplicate_pair = 0

    equity = INITIAL_CAPITAL
    daily_records = []

    for date in calendar:
        equity_before = equity

        # Enter at today's Open, based on previously generated signals.
        # Positions exiting today still occupy capacity at the Open.
        for trade in entries.get(date, []):
            if len(active) >= MAX_OPEN_TRADES:
                skipped_capacity += 1
                continue

            if any(t["_pair"] == trade["_pair"] for t in active):
                skipped_duplicate_pair += 1
                continue

            trade["_capital"] = (
                equity_before * ALLOCATION_PER_TRADE
            )
            active.append(trade)
            executed.append(trade)

        dollar_pnl = 0.0

        for trade in active:
            unit_pnl = trade["_daily_pnl_per_dollar"].get(date, 0.0)
            dollar_pnl += trade["_capital"] * unit_pnl

        equity = equity_before + dollar_pnl

        if equity <= 0:
            raise RuntimeError(
                f"{strategy_name}: portfolio equity reached zero "
                f"or below on {date.date()}."
            )

        daily_records.append(
            {
                "date": date,
                "equity": equity,
                "daily_return": dollar_pnl / equity_before,
                "daily_pnl": dollar_pnl,
                "active_trades": len(active),
            }
        )

        # An exit occurs at today's Close.
        active = [
            t for t in active
            if t["_exit_date"] > date
        ]

    daily = pd.DataFrame(daily_records).set_index("date")

    return {
        "name": strategy_name,
        "daily": daily,
        "executed": executed,
        "candidates": len(candidates),
        "skipped_capacity": skipped_capacity,
        "skipped_duplicate_pair": skipped_duplicate_pair,
    }


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(result):
    daily = result["daily"]
    returns = daily["daily_return"].astype(float)

    equity_curve = np.r_[INITIAL_CAPITAL, daily["equity"].to_numpy()]
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / peaks - 1.0

    total_return = daily["equity"].iloc[-1] / INITIAL_CAPITAL - 1.0
    n_days = len(returns)

    cagr = (
        (daily["equity"].iloc[-1] / INITIAL_CAPITAL)
        ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
        if n_days > 0 else np.nan
    )

    std = returns.std(ddof=1)
    sharpe = (
        np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / std
        if std and np.isfinite(std) else np.nan
    )

    downside_rms = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    sortino = (
        np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / downside_rms
        if downside_rms > 0 else np.nan
    )

    trades = result["executed"]
    trade_returns = pd.Series(
        [t["_trade_return"] for t in trades],
        dtype=float,
    )

    if len(trade_returns):
        gains = trade_returns[trade_returns > 0].sum()
        losses = -trade_returns[trade_returns < 0].sum()
        profit_factor = gains / losses if losses > 0 else np.inf
        q05 = trade_returns.quantile(0.05)
        expected_shortfall = trade_returns[
            trade_returns <= q05
        ].mean()
        worst_trade = trade_returns.min()
        win_rate = (trade_returns > 0).mean()
        median_trade = trade_returns.median()
    else:
        profit_factor = np.nan
        q05 = np.nan
        expected_shortfall = np.nan
        worst_trade = np.nan
        win_rate = np.nan
        median_trade = np.nan

    max_dd = drawdowns.min()

    metrics = {
        "OOS candidate events": result["candidates"],
        "Executed trades": len(trades),
        "Skipped: capacity full": result["skipped_capacity"],
        "Skipped: pair already open": result["skipped_duplicate_pair"],
        "Total portfolio return": total_return,
        "Annualized CAGR": cagr,
        "Maximum drawdown": max_dd,
        "Annualized Sharpe (rf=0)": sharpe,
        "Annualized Sortino (target=0)": sortino,
        "Calmar ratio": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "Worst daily portfolio return": returns.min(),
        "Win rate per executed trade": win_rate,
        "Median trade return": median_trade,
        "Worst trade return": worst_trade,
        "5th percentile trade return": q05,
        "Mean return in worst 5% tail": expected_shortfall,
        "Trade profit factor": profit_factor,
        "Final equity": daily["equity"].iloc[-1],
    }

    return metrics


def yearly_report(result):
    daily = result["daily"].copy()
    daily["year"] = daily.index.year

    rows = []

    for year, group in daily.groupby("year"):
        r = group["daily_return"]
        wealth = (1.0 + r).cumprod()
        dd = (wealth / wealth.cummax() - 1.0).min()

        std = r.std(ddof=1)
        sharpe = (
            np.sqrt(TRADING_DAYS_PER_YEAR) * r.mean() / std
            if std and np.isfinite(std) else np.nan
        )

        trades_in_year = sum(
            pd.Timestamp(t["exit_date"]).year == year
            for t in result["executed"]
        )

        rows.append(
            {
                "year": year,
                "trades exited": trades_in_year,
                "portfolio return": wealth.iloc[-1] - 1.0,
                "max drawdown within year": dd,
                "annualized Sharpe": sharpe,
                "worst daily return": r.min(),
            }
        )

    return pd.DataFrame(rows)


def print_metrics(result):
    print("\n" + "=" * 72)
    print(result["name"])
    print("=" * 72)

    metrics = calculate_metrics(result)

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            if any(
                word in key.lower()
                for word in ("return", "drawdown", "rate", "percentile", "tail")
            ):
                print(f"{key:38s}: {value: .2%}")
            elif key == "Final equity":
                print(f"{key:38s}: ${value:,.2f}")
            else:
                print(f"{key:38s}: {value: .4f}")
        else:
            print(f"{key:38s}: {value}")

    print("\nPerformance by calendar year:")
    yearly = yearly_report(result)
    if yearly.empty:
        print("No yearly results.")
    else:
        print(yearly.to_string(
            index=False,
            formatters={
                "portfolio return": "{:.2%}".format,
                "max drawdown within year": "{:.2%}".format,
                "annualized Sharpe": "{:.3f}".format,
                "worst daily return": "{:.2%}".format,
            },
        ))


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(EVENT_FILE)

    required = {
        "sample_type",
        "threshold",
        "horizon",
        "ticker_a",
        "ticker_b",
        "signal_date",
        "entry_date",
        "exit_date",
        "signal_type",
        "hedge_ratio",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing required event columns: {sorted(missing)}")

    events = events[
        events["sample_type"].astype(str).str.upper().eq("OOS")
        & np.isclose(pd.to_numeric(events["threshold"]), THRESHOLD)
        & pd.to_numeric(events["horizon"]).eq(HORIZON)
    ].copy()

    if events.empty:
        raise RuntimeError(
            "No OOS events found for the selected threshold and horizon."
        )

    for c in ("signal_date", "entry_date", "exit_date"):
        events[c] = pd.to_datetime(events[c]).dt.normalize()

    events["ticker_a"] = events["ticker_a"].astype(str).str.upper()
    events["ticker_b"] = events["ticker_b"].astype(str).str.upper()
    events["signal_type"] = events["signal_type"].astype(str).str.upper()

    tickers = set(events["ticker_a"]) | set(events["ticker_b"])
    prices = load_all_prices(tickers)

    reconstructed = []
    invalid = []

    for idx, row in events.reset_index(drop=True).iterrows():
        try:
            trade = reconstruct_trade(row, prices)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reconstruct event {idx}: "
                f"{row['ticker_a']}/{row['ticker_b']}, "
                f"{row['entry_date']} to {row['exit_date']}: {exc}"
            ) from exc

        if trade is None:
            invalid.append(idx)
        else:
            reconstructed.append(trade)

    if not reconstructed:
        raise RuntimeError(
            "No events could be reconstructed. Check raw price files "
            "and entry/exit dates."
        )

    # One common calendar for both strategies.
    start = min(t["_entry_date"] for t in reconstructed)
    end = max(t["_exit_date"] for t in reconstructed)

    date_set = set()
    for frame in prices.values():
        date_set.update(frame.index[(frame.index >= start) & (frame.index <= end)])

    calendar = sorted(date_set)

    if not calendar:
        raise RuntimeError("Could not build the OOS trading calendar.")

    bidirectional_trades = reconstructed
    long_a_trades = [
        t for t in reconstructed if t["signal_type"] == "LONG_A"
    ]

    print("OOS DIRECTION STRATEGY COMPARISON")
    print(f"Threshold: {THRESHOLD}")
    print(f"Horizon: {HORIZON} trading-session convention from source backtest")
    print(f"Original OOS events: {len(events)}")
    print(f"Events reconstructed: {len(reconstructed)}")
    print(f"Events skipped due to missing price dates: {len(invalid)}")
    print(f"Calendar dates: {len(calendar)}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Maximum concurrent trades: {MAX_OPEN_TRADES}")
    print(f"Transaction cost assumption: {TRANSACTION_COST_BPS:.1f} bps per dollar of turnover")
    print(f"Annual short borrow assumption: {ANNUAL_SHORT_BORROW_RATE:.1%}")

    # Identical portfolio rules, price data, and calendar.
    both = simulate_portfolio(
        bidirectional_trades,
        calendar,
        "A. Original bidirectional strategy",
    )

    only_a = simulate_portfolio(
        long_a_trades,
        calendar,
        "B. LONG_A only",
    )

    print_metrics(both)
    print_metrics(only_a)

    comparison = pd.DataFrame(
        {
            "Bidirectional": calculate_metrics(both),
            "LONG_A only": calculate_metrics(only_a),
        }
    )
    comparison.to_csv(OUTPUT_DIR / "oos_direction_comparison.csv")

    for result, filename in [
        (both, "oos_bidirectional_equity.csv"),
        (only_a, "oos_long_a_equity.csv"),
    ]:
        result["daily"].to_csv(OUTPUT_DIR / filename)

    print("\nSaved reports to:")
    print(OUTPUT_DIR / "oos_direction_comparison.csv")
    print(OUTPUT_DIR / "oos_bidirectional_equity.csv")
    print(OUTPUT_DIR / "oos_long_a_equity.csv")


if __name__ == "__main__":
    main()