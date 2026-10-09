
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd


ENGINE_PATH = Path("scripts/evaluate_oos_direction_strategies.py")
OUTPUT_DIR = Path("artifacts/reports")

BASE_COST_BPS = 5.0
BASE_BORROW_RATE = 0.03
ALLOCATION = 0.05
MAX_TRADES = 20


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "oos_concentration_engine", ENGINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ENGINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def portfolio_summary(result):
    daily = result["daily"]
    returns = daily["daily_return"].astype(float)
    equity = daily["equity"].astype(float)

    total_return = (
        float(equity.iloc[-1]) / 100_000.0 - 1.0
    ) * 100.0

    wealth = np.r_[1.0, np.cumprod(1.0 + returns.to_numpy())]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0

    std = returns.std(ddof=1)
    sharpe = (
        returns.mean() / std * np.sqrt(252)
        if pd.notna(std) and std > 0
        else np.nan
    )

    return {
        "strategy": result["name"],
        "return_pct": total_return,
        "max_drawdown_pct": float(drawdown.min()) * 100.0,
        "sharpe": float(sharpe),
        "executed_trades": len(result["executed"]),
        "skipped_capacity": result["skipped_capacity"],
        "skipped_duplicate_pair": result["skipped_duplicate_pair"],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = load_engine()
    engine.TRANSACTION_COST_BPS = BASE_COST_BPS
    engine.ANNUAL_SHORT_BORROW_RATE = BASE_BORROW_RATE
    engine.MAX_OPEN_TRADES = MAX_TRADES
    engine.ALLOCATION_PER_TRADE = ALLOCATION
    engine.INITIAL_CAPITAL = 100_000.0

    events = pd.read_parquet(engine.EVENT_FILE)
    events = events[
        events["sample_type"].astype(str).str.upper().eq("OOS")
        & np.isclose(pd.to_numeric(events["threshold"]), engine.THRESHOLD)
        & pd.to_numeric(events["horizon"]).eq(engine.HORIZON)
    ].copy()

    for col in ("signal_date", "entry_date", "exit_date"):
        events[col] = pd.to_datetime(events[col]).dt.normalize()

    for col in ("ticker_a", "ticker_b", "signal_type"):
        events[col] = events[col].astype(str).str.upper()

    prices = engine.load_all_prices(
        set(events["ticker_a"]) | set(events["ticker_b"])
    )

    trades = []
    invalid = 0

    for event_id, (_, row) in enumerate(events.iterrows()):
        trade = engine.reconstruct_trade(row, prices)
        if trade is None:
            invalid += 1
            continue

        trade["_stress_id"] = event_id
        trades.append(trade)

    if not trades:
        raise RuntimeError("No valid trades were reconstructed.")

    start = min(t["_entry_date"] for t in trades)
    end = max(t["_exit_date"] for t in trades)
    dates = set()

    for frame in prices.values():
        dates.update(frame.index[(frame.index >= start) & (frame.index <= end)])

    calendar = sorted(dates)
    if not calendar:
        raise RuntimeError("No trading calendar available.")

    def run(candidate_trades, name):
        return engine.simulate_portfolio(
            candidate_trades, calendar, name
        )

    bidirectional = run(trades, "Bidirectional")
    long_a = run(
        [t for t in trades if t["signal_type"] == "LONG_A"],
        "LONG_A only",
    )
    long_b = run(
        [t for t in trades if t["signal_type"] == "LONG_B"],
        "LONG_B only",
    )

    summaries = [
        portfolio_summary(bidirectional),
        portfolio_summary(long_a),
        portfolio_summary(long_b),
    ]

    # Rank executed baseline trades by realized dollar contribution.
    executed = bidirectional["executed"]
    contributions = []

    for trade in executed:
        contributions.append({
            "stress_id": trade["_stress_id"],
            "ticker_a": trade["ticker_a"],
            "ticker_b": trade["ticker_b"],
            "signal_type": trade["signal_type"],
            "entry_date": trade["_entry_date"],
            "exit_date": trade["_exit_date"],
            "capital_at_entry": trade["_capital"],
            "unit_trade_return": trade["_trade_return"],
            "dollar_contribution": (
                trade["_capital"] * trade["_trade_return"]
            ),
        })

    contributions = pd.DataFrame(contributions).sort_values(
        "dollar_contribution", ascending=False
    )

    contributions.to_csv(
        OUTPUT_DIR / "oos_trade_contributions.csv", index=False
    )

    # This is an ex-post fragility diagnostic, not a tradable rule.
    for k in (1, 3, 5):
        removed_ids = set(
            contributions.head(k)["stress_id"].astype(int)
        )
        remaining = [
            t for t in trades
            if t["_stress_id"] not in removed_ids
        ]

        result = run(
            remaining,
            f"Bidirectional excluding top {k} winners",
        )
        summaries.append(portfolio_summary(result))

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "oos_concentration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    positive_contribution = contributions[
        contributions["dollar_contribution"] > 0
    ]
    total_positive = positive_contribution["dollar_contribution"].sum()

    print("=" * 72)
    print("OOS DIRECTION AND WINNER-CONCENTRATION TEST")
    print("=" * 72)
    print(f"Valid trades: {len(trades)}")
    print(f"Invalid trades skipped: {invalid}")
    print(
        summary_df.to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nTOP WINNERS BY REALIZED DOLLAR CONTRIBUTION")
    print(
        contributions.head(10).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nWINNER CONCENTRATION")
    if total_positive > 0:
        for k in (1, 3, 5):
            share = (
                positive_contribution.head(k)["dollar_contribution"].sum()
                / total_positive
                * 100.0
            )
            print(f"Top {k} winners / total positive contributions: {share:.1f}%")
    else:
        print("No positive dollar contributions to measure.")

    print("\nSaved:")
    print(summary_path)
    print(OUTPUT_DIR / "oos_trade_contributions.csv")


if __name__ == "__main__":
    main()