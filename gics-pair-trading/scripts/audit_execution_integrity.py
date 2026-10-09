from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKTEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pair_event_backtest_tier1.parquet"
)

CLOSE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "close_prices.parquet"
)

OPEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "open_prices.parquet"
)


def main():
    print("=" * 80)
    print("EXECUTION INTEGRITY AUDIT")
    print("=" * 80)

    trades = pd.read_parquet(BACKTEST_PATH)
    close = pd.read_parquet(CLOSE_PATH)
    open_prices = pd.read_parquet(OPEN_PATH)

    trades["signal_date"] = pd.to_datetime(trades["signal_date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    close.index = pd.to_datetime(close.index)
    open_prices.index = pd.to_datetime(open_prices.index)

    print(f"Backtest rows: {len(trades):,}")
    print(f"Unique signal dates: {trades['signal_date'].nunique():,}")
    print()

    errors = []

    # ------------------------------------------------------------------
    # 1. Entry must be strictly after signal
    # ------------------------------------------------------------------

    bad_order = trades[
        trades["entry_date"] <= trades["signal_date"]
    ]

    print(
        f"[1] Entry after signal: "
        f"{len(bad_order)} violations"
    )

    if len(bad_order):
        errors.append("entry_date <= signal_date")

    # ------------------------------------------------------------------
    # 2. Entry must be the next valid pair trading date
    # ------------------------------------------------------------------

    next_date_errors = 0

    for _, row in trades.iterrows():

        pair_dates = close[
            [row["ticker_a"], row["ticker_b"]]
        ].dropna().index

        signal_date = row["signal_date"]

        if signal_date not in pair_dates:
            next_date_errors += 1
            continue

        pos = pair_dates.get_loc(signal_date)

        if pos + 1 >= len(pair_dates):
            next_date_errors += 1
            continue

        expected_entry = pair_dates[pos + 1]

        if row["entry_date"] != expected_entry:
            next_date_errors += 1

    print(
        f"[2] T+1 pair-valid entry date: "
        f"{next_date_errors} violations"
    )

    if next_date_errors:
        errors.append("entry_date is not next valid pair date")

    # ------------------------------------------------------------------
    # 3. Entry prices must equal Open
    # ------------------------------------------------------------------

    entry_price_errors = 0

    for _, row in trades.iterrows():

        a_open = open_prices.loc[
            row["entry_date"],
            row["ticker_a"],
        ]

        b_open = open_prices.loc[
            row["entry_date"],
            row["ticker_b"],
        ]

        if not np.isclose(
            row["entry_open_a"],
            a_open,
            rtol=1e-10,
            atol=1e-10,
        ):
            entry_price_errors += 1

        if not np.isclose(
            row["entry_open_b"],
            b_open,
            rtol=1e-10,
            atol=1e-10,
        ):
            entry_price_errors += 1

    print(
        f"[3] Entry prices equal Open: "
        f"{entry_price_errors} violations"
    )

    if entry_price_errors:
        errors.append("entry price != open price")

    # ------------------------------------------------------------------
    # 4. Exit date must equal signal + horizon
    #    using pair-valid dates
    # ------------------------------------------------------------------

    exit_date_errors = 0

    for _, row in trades.iterrows():

        pair_dates = close[
            [row["ticker_a"], row["ticker_b"]]
        ].dropna().index

        signal_date = row["signal_date"]

        if signal_date not in pair_dates:
            exit_date_errors += 1
            continue

        signal_pos = pair_dates.get_loc(signal_date)

        expected_pos = (
            signal_pos + int(row["horizon"])
        )

        if expected_pos >= len(pair_dates):
            exit_date_errors += 1
            continue

        expected_exit = pair_dates[expected_pos]

        if row["exit_date"] != expected_exit:
            exit_date_errors += 1

    print(
        f"[4] Exit date = T+H: "
        f"{exit_date_errors} violations"
    )

    if exit_date_errors:
        errors.append("exit_date != T+H")

    # ------------------------------------------------------------------
    # 5. Exit prices must equal Close
    # ------------------------------------------------------------------

    exit_price_errors = 0

    for _, row in trades.iterrows():

        a_close = close.loc[
            row["exit_date"],
            row["ticker_a"],
        ]

        b_close = close.loc[
            row["exit_date"],
            row["ticker_b"],
        ]

        if not np.isclose(
            row["exit_close_a"],
            a_close,
            rtol=1e-10,
            atol=1e-10,
        ):
            exit_price_errors += 1

        if not np.isclose(
            row["exit_close_b"],
            b_close,
            rtol=1e-10,
            atol=1e-10,
        ):
            exit_price_errors += 1

    print(
        f"[5] Exit prices equal Close: "
        f"{exit_price_errors} violations"
    )

    if exit_price_errors:
        errors.append("exit price != close price")

    # ------------------------------------------------------------------
    # 6. No missing execution prices
    # ------------------------------------------------------------------

    missing_execution = trades[
        trades[
            [
                "entry_open_a",
                "entry_open_b",
                "exit_close_a",
                "exit_close_b",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    print(
        f"[6] Missing execution prices: "
        f"{len(missing_execution)} violations"
    )

    if len(missing_execution):
        errors.append("missing execution prices")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print()
    print("=" * 80)

    if errors:
        print("EXECUTION AUDIT: FAILED")
        print()
        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("EXECUTION AUDIT: PASSED")
    print()
    print("Execution model verified:")
    print("  Signal:       T Close")
    print("  Entry:        T+1 Open")
    print("  Exit:         T+H Close")
    print("  Missing-data: pair-specific valid-date calendar")
    print("=" * 80)


if __name__ == "__main__":
    main()