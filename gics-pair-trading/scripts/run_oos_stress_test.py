
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ENGINE_PATH = Path("scripts/evaluate_oos_direction_strategies.py")
OUTPUT_DIR = Path("artifacts/reports")

# Full-factorial stress grid.
TRANSACTION_COSTS_BPS = [5.0, 10.0, 20.0, 30.0]
BORROW_RATES = [0.03, 0.05, 0.10, 0.20]
ALLOCATIONS = [0.025, 0.05, 0.10]

# Keep the capacity rule unchanged.
MAX_OPEN_TRADES = 20


def load_engine():
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {ENGINE_PATH}. "
            "Run this script from the project root directory."
        )

    spec = importlib.util.spec_from_file_location(
        "oos_backtest_engine", ENGINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the original backtest script.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_events(engine):
    events = pd.read_parquet(engine.EVENT_FILE)

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
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    events = events[
        events["sample_type"].astype(str).str.upper().eq("OOS")
        & np.isclose(pd.to_numeric(events["threshold"]), engine.THRESHOLD)
        & pd.to_numeric(events["horizon"]).eq(engine.HORIZON)
    ].copy()

    if events.empty:
        raise RuntimeError("No matching OOS events found.")

    for col in ("signal_date", "entry_date", "exit_date"):
        events[col] = pd.to_datetime(events[col]).dt.normalize()

    for col in ("ticker_a", "ticker_b", "signal_type"):
        events[col] = events[col].astype(str).str.upper()

    return events.reset_index(drop=True)


def reconstruct_valid_events(engine, events, prices):
    valid_rows = []
    invalid_count = 0

    for _, row in events.iterrows():
        trade = engine.reconstruct_trade(row, prices)
        if trade is None:
            invalid_count += 1
        else:
            valid_rows.append(row)

    if not valid_rows:
        raise RuntimeError("No valid OOS trades could be reconstructed.")

    return valid_rows, invalid_count


def build_calendar(prices, trades):
    start = min(t["_entry_date"] for t in trades)
    end = max(t["_exit_date"] for t in trades)

    date_set = set()
    for frame in prices.values():
        date_set.update(
            frame.index[(frame.index >= start) & (frame.index <= end)]
        )

    calendar = sorted(date_set)
    if not calendar:
        raise RuntimeError("Could not build the trading calendar.")

    return calendar


def yearly_stats(result, scenario):
    daily = result["daily"]
    rows = []

    for year, group in daily.groupby(daily.index.year):
        returns = group["daily_return"].astype(float).to_numpy()

        wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
        peaks = np.maximum.accumulate(wealth)
        drawdowns = wealth / peaks - 1.0

        std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        sharpe = (
            np.mean(returns) / std * np.sqrt(252)
            if std > 0
            else np.nan
        )

        rows.append({
            **scenario,
            "year": int(year),
            "year_return_pct": (wealth[-1] - 1.0) * 100.0,
            "year_max_drawdown_pct": float(drawdowns.min()) * 100.0,
            "annualized_sharpe": float(sharpe),
            "worst_daily_return_pct": float(returns.min()) * 100.0,
        })

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = load_engine()
    events = prepare_events(engine)

    tickers = set(events["ticker_a"]) | set(events["ticker_b"])
    prices = engine.load_all_prices(tickers)

    # Determine valid events once, using the original baseline assumptions.
    valid_rows, invalid_count = reconstruct_valid_events(
        engine, events, prices
    )

    engine.TRANSACTION_COST_BPS = 5.0
    engine.ANNUAL_SHORT_BORROW_RATE = 0.03

    baseline_trades = [
        engine.reconstruct_trade(row, prices)
        for row in valid_rows
    ]
    calendar = build_calendar(prices, baseline_trades)

    print("=" * 72)
    print("OOS PORTFOLIO STRESS TEST")
    print("=" * 72)
    print(f"Original OOS events: {len(events)}")
    print(f"Valid events: {len(valid_rows)}")
    print(f"Invalid events skipped: {invalid_count}")
    print(f"Calendar dates: {len(calendar)}")
    print(f"Maximum concurrent trades: {MAX_OPEN_TRADES}")
    print("Allocation means gross capital allocated per trade.")
    print("Maximum gross allocation at full capacity is allocation x 20.")
    print()

    result_rows = []
    annual_rows = []

    total_scenarios = (
        len(TRANSACTION_COSTS_BPS)
        * len(BORROW_RATES)
        * len(ALLOCATIONS)
    )
    scenario_number = 0

    for cost_bps in TRANSACTION_COSTS_BPS:
        for borrow_rate in BORROW_RATES:
            for allocation in ALLOCATIONS:
                scenario_number += 1

                # These parameters are used during trade reconstruction.
                engine.TRANSACTION_COST_BPS = cost_bps
                engine.ANNUAL_SHORT_BORROW_RATE = borrow_rate

                trades = [
                    engine.reconstruct_trade(row, prices)
                    for row in valid_rows
                ]
                trades = [t for t in trades if t is not None]

                # Set portfolio sizing independently of capacity.
                engine.MAX_OPEN_TRADES = MAX_OPEN_TRADES
                engine.ALLOCATION_PER_TRADE = allocation

                scenario = {
                    "transaction_cost_bps": cost_bps,
                    "annual_borrow_rate_pct": borrow_rate * 100.0,
                    "allocation_per_trade_pct": allocation * 100.0,
                    "max_gross_allocation_pct": (
                        MAX_OPEN_TRADES * allocation * 100.0
                    ),
                }

                both = engine.simulate_portfolio(
                    trades, calendar, "Bidirectional"
                )
                only_a_trades = [
                    t for t in trades if t["signal_type"] == "LONG_A"
                ]
                only_a = engine.simulate_portfolio(
                    only_a_trades, calendar, "LONG_A only"
                )

                for result in (both, only_a):
                    metrics = engine.calculate_metrics(result)
                    final_equity = float(
                        result["daily"]["equity"].iloc[-1]
                    )

                    row = {
                        **scenario,
                        "strategy": result["name"],
                        "final_equity": final_equity,
                        "portfolio_return_pct": (
                            final_equity / engine.INITIAL_CAPITAL - 1.0
                        ) * 100.0,
                        "executed_trades": len(result["executed"]),
                        "skipped_capacity": result["skipped_capacity"],
                        "skipped_duplicate_pair": (
                            result["skipped_duplicate_pair"]
                        ),
                    }

                    for key, value in metrics.items():
                        row[f"metric_{key}"] = value

                    result_rows.append(row)
                    annual_rows.extend(
                        yearly_stats(result, scenario | {
                            "strategy": result["name"]
                        })
                    )

                print(
                    f"[{scenario_number:02d}/{total_scenarios}] "
                    f"cost={cost_bps:>4.0f} bps, "
                    f"borrow={borrow_rate:>4.0%}, "
                    f"allocation={allocation:>4.1%} done"
                )

    results = pd.DataFrame(result_rows)
    yearly = pd.DataFrame(annual_rows)

    results_path = OUTPUT_DIR / "oos_stress_test_results.csv"
    yearly_path = OUTPUT_DIR / "oos_stress_test_yearly.csv"

    results.to_csv(results_path, index=False)
    yearly.to_csv(yearly_path, index=False)

    # Show the most useful columns first.
    display_cols = [
        "strategy",
        "transaction_cost_bps",
        "annual_borrow_rate_pct",
        "allocation_per_trade_pct",
        "max_gross_allocation_pct",
        "portfolio_return_pct",
        "executed_trades",
    ]

    print("\n" + "=" * 72)
    print("OVERALL RESULTS")
    print("=" * 72)
    print(
        results[display_cols]
        .sort_values(
            ["strategy", "transaction_cost_bps",
             "annual_borrow_rate_pct", "allocation_per_trade_pct"]
        )
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    # Locate baseline and combined severe-cost scenarios.
    baseline = results[
        np.isclose(results["transaction_cost_bps"], 5.0)
        & np.isclose(results["annual_borrow_rate_pct"], 3.0)
        & np.isclose(results["allocation_per_trade_pct"], 5.0)
    ]

    severe = results[
        np.isclose(results["transaction_cost_bps"], 30.0)
        & np.isclose(results["annual_borrow_rate_pct"], 20.0)
        & np.isclose(results["allocation_per_trade_pct"], 10.0)
    ]

    print("\nBASELINE vs COMBINED STRESS")
    comparison_cols = [
        "strategy",
        "portfolio_return_pct",
        "final_equity",
    ]
    print("\nBaseline: 5 bps cost / 3% borrow / 5% allocation")
    print(baseline[comparison_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    print("\nCombined stress: 30 bps cost / 20% borrow / 10% allocation")
    print(severe[comparison_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\nSaved:")
    print(results_path)
    print(yearly_path)


if __name__ == "__main__":
    main()