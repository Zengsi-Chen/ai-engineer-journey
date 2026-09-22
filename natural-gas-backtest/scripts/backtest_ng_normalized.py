from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "ng_strategy_execution_2021_2026.csv"

SUMMARY_FILE = DATA_DIR / "ng_normalized_backtest_summary.csv"
ANNUAL_FILE = DATA_DIR / "ng_normalized_annual_returns.csv"
TRADES_FILE = DATA_DIR / "ng_normalized_trade_events.csv"
EQUITY_FILE = DATA_DIR / "ng_normalized_equity_curves.csv"

START_DATE = pd.Timestamp("2021-09-17")
END_DATE = pd.Timestamp("2026-09-17")

INITIAL_CAPITAL = 10_000.0
TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# LOAD
# =============================================================================

def load_data():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date", "execution_date"]
    )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["execution_date"] = pd.to_datetime(
        df["execution_date"]
    ).dt.normalize()

    df = df.sort_values("date").reset_index(drop=True)

    df = df[
        (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
    ].copy()

    numeric_cols = [
        "signal_raw_settlement",
        "back_adjusted",
        "dma20",
        "std20",
        "bb_upper",
        "bb_lower",
        "dma50",
        "dma200",
        "execution_open",
        "execution_high",
        "execution_low",
        "execution_close",
        "execution_volume",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df.reset_index(drop=True)


# =============================================================================
# STRATEGY SIGNALS
# =============================================================================

def get_signal(df, strategy):

    signal = pd.Series(
        "NONE",
        index=df.index
    )

    if strategy == "Buy & Hold":

        signal.iloc[0] = "BUY"

    elif strategy == "50/200 DMA":

        signal[
            df["dma50"] > df["dma200"]
        ] = "BUY"

        signal[
            df["dma50"] <= df["dma200"]
        ] = "SELL"

    elif strategy == "Bollinger Mean Reversion":

        signal[
            df["mr_entry_signal"].astype(bool)
        ] = "BUY"

        signal[
            df["mr_exit_signal"].astype(bool)
        ] = "SELL"

    elif strategy == "$2 Contrarian":

        signal[
            df["under_2_entry_signal"].astype(bool)
        ] = "BUY"

        signal[
            df["under_2_exit_signal"].astype(bool)
        ] = "SELL"

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    return signal


# =============================================================================
# EVENT GENERATION
# =============================================================================

def build_events(df, strategy):

    signal = get_signal(
        df,
        strategy
    )

    events = []

    position = 0

    for i in range(len(df)):

        signal_date = df.loc[i, "date"]

        action = signal.iloc[i]

        execution_date = df.loc[i, "execution_date"]

        # No T+1 session available.
        if pd.isna(execution_date):
            continue

        # Prevent look-ahead.
        if execution_date <= signal_date:

            raise ValueError(
                f"Look-ahead detected: "
                f"{signal_date.date()} -> "
                f"{execution_date.date()}"
            )

        # Cannot execute beyond formal study end.
        if execution_date > END_DATE:
            continue

        execution_price = df.loc[
            i,
            "execution_open"
        ]

        execution_contract = df.loc[
            i,
            "execution_contract"
        ]

        if pd.isna(execution_price):
            continue

        # -------------------------------------------------------------
        # BUY
        # -------------------------------------------------------------

        if action == "BUY" and position == 0:

            events.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "side": "BUY",
                    "execution_contract":
                        execution_contract,
                    "execution_price":
                        execution_price,
                }
            )

            position = 1

        # -------------------------------------------------------------
        # SELL
        # -------------------------------------------------------------

        elif action == "SELL" and position == 1:

            events.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "side": "SELL",
                    "execution_contract":
                        execution_contract,
                    "execution_price":
                        execution_price,
                }
            )

            position = 0

    events = pd.DataFrame(events)

    if not events.empty:

        events = (
            events
            .sort_values("execution_date")
            .reset_index(drop=True)
        )

        if events["execution_date"].duplicated().any():

            raise ValueError(
                "Multiple strategy events "
                "occur on the same execution date."
            )

    return events


# =============================================================================
# MARKET TIMELINE
# =============================================================================

def build_market_timeline(df):

    """
    Each row represents an official settlement observation.

    Signal:
        current settlement

    Execution:
        next settlement-chain trading date,
        actual active contract Open

    Market mark:
        current settlement

    This allows us to explicitly model:

        flat
          ↓
        T signal
          ↓
        T+1 Open
          ↓
        T+1 Settlement
          ↓
        subsequent settlements
          ↓
        roll gaps
          ↓
        T+1 exit Open
    """

    market = df[
        [
            "date",
            "signal_contract",
            "signal_raw_settlement",
            "execution_date",
            "execution_contract",
            "execution_open",
            "contract_changed",
        ]
    ].copy()

    market = market.rename(
        columns={
            "signal_contract":
                "active_contract",
            "signal_raw_settlement":
                "settlement",
        }
    )

    market = (
        market
        .sort_values("date")
        .reset_index(drop=True)
    )

    market["previous_settlement"] = (
        market["settlement"].shift(1)
    )

    market["previous_contract"] = (
        market["active_contract"].shift(1)
    )

    market["roll_today"] = (
        market["active_contract"]
        != market["previous_contract"]
    )

    return market


# =============================================================================
# EVENT-DRIVEN ENGINE
# =============================================================================

def run_strategy(df, strategy):

    events = build_events(
        df,
        strategy
    )

    market = build_market_timeline(
        df
    )

    event_lookup = {}

    for _, event in events.iterrows():

        date = event["execution_date"]

        event_lookup[date] = event

    equity = INITIAL_CAPITAL

    position = 0

    entry_price = np.nan
    entry_date = pd.NaT
    entry_contract = None

    equity_rows = []
    trade_rows = []

    previous_settlement = np.nan

    for _, row in market.iterrows():

        date = row["date"]

        settlement = row["settlement"]

        active_contract = row["active_contract"]

        previous_contract = row["previous_contract"]

        event = event_lookup.get(date)

        action = "NONE"
        execution_price = np.nan

        daily_return = 0.0

        # =============================================================
        # EVENT AT TODAY'S OPEN
        # =============================================================

        if event is not None:

            action = event["side"]

            execution_price = (
                event["execution_price"]
            )

            # ---------------------------------------------------------
            # BUY
            # ---------------------------------------------------------

            if action == "BUY":

                if position != 0:

                    raise ValueError(
                        f"BUY while already long: "
                        f"{date.date()}"
                    )

                # We were flat before today's open.
                #
                # Therefore:
                #
                # Open -> Settlement
                #
                daily_return = (
                    settlement
                    / execution_price
                    - 1.0
                )

                position = 1

                entry_price = execution_price
                entry_date = date
                entry_contract = (
                    event["execution_contract"]
                )

            # ---------------------------------------------------------
            # SELL
            # ---------------------------------------------------------

            elif action == "SELL":

                if position != 1:

                    raise ValueError(
                        f"SELL while flat: "
                        f"{date.date()}"
                    )

                if pd.isna(previous_settlement):

                    raise ValueError(
                        "SELL has no previous settlement."
                    )

                # Position was held through the previous settlement.
                #
                # Previous settlement -> today's execution Open
                #
                daily_return = (
                    execution_price
                    / previous_settlement
                    - 1.0
                )

                trade_return = (
                    execution_price
                    / entry_price
                    - 1.0
                )

                trade_rows.append(
                    {
                        "strategy": strategy,
                        "entry_date": entry_date,
                        "entry_contract":
                            entry_contract,
                        "entry_price":
                            entry_price,
                        "exit_date": date,
                        "exit_contract":
                            event["execution_contract"],
                        "exit_price":
                            execution_price,
                        "trade_return":
                            trade_return,
                        "holding_days":
                            (date - entry_date).days,
                        "status": "CLOSED",
                    }
                )

                position = 0

                entry_price = np.nan
                entry_date = pd.NaT
                entry_contract = None

        # =============================================================
        # NO EXECUTION EVENT
        # =============================================================

        else:

            if position == 1:

                if pd.isna(previous_settlement):

                    daily_return = 0.0

                else:

                    # -------------------------------------------------
                    # Normal holding day
                    #
                    # settlement_t /
                    # settlement_(t-1)
                    #
                    # -------------------------------------------------
                    daily_return = (
                        settlement
                        / previous_settlement
                        - 1.0
                    )

            else:

                daily_return = 0.0

        # =============================================================
        # EQUITY UPDATE
        # =============================================================

        equity *= (
            1.0 + daily_return
        )

        equity_rows.append(
            {
                "date": date,
                "strategy": strategy,
                "equity": equity,
                "position": position,
                "active_contract":
                    active_contract,
                "settlement":
                    settlement,
                "previous_contract":
                    previous_contract,
                "roll_today":
                    row["roll_today"],
                "action":
                    action,
                "execution_price":
                    execution_price,
                "daily_return":
                    daily_return,
            }
        )

        previous_settlement = settlement

    # =============================================================
    # OPEN POSITION AT END
    # =============================================================

    if position == 1:

        final_settlement = (
            market["settlement"].iloc[-1]
        )

        marked_trade_return = (
            final_settlement
            / entry_price
            - 1.0
        )

        trade_rows.append(
            {
                "strategy": strategy,
                "entry_date": entry_date,
                "entry_contract":
                    entry_contract,
                "entry_price":
                    entry_price,
                "exit_date":
                    pd.NaT,
                "exit_contract":
                    None,
                "exit_price":
                    np.nan,
                "trade_return":
                    marked_trade_return,
                "holding_days":
                    (
                        END_DATE
                        - entry_date
                    ).days,
                "status":
                    "OPEN_AT_END",
            }
        )

    equity_df = pd.DataFrame(
        equity_rows
    )

    trades_df = pd.DataFrame(
        trade_rows
    )

    return (
        equity_df,
        trades_df,
        events
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    equity_df,
    trades_df,
    strategy
):

    equity = equity_df["equity"]

    final_value = float(
        equity.iloc[-1]
    )

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1.0
    )

    years = (
        equity_df["date"].iloc[-1]
        - equity_df["date"].iloc[0]
    ).days / 365.25

    cagr = (
        (final_value / INITIAL_CAPITAL)
        ** (1.0 / years)
        - 1.0
    )

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    daily_returns = (
        equity_df["daily_return"]
    )

    daily_std = daily_returns.std(
        ddof=1
    )

    if daily_std > 0:

        sharpe = (
            daily_returns.mean()
            / daily_std
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )

    else:

        sharpe = np.nan

    if trades_df.empty:

        completed = pd.DataFrame()

        open_trades = 0

    else:

        completed = trades_df[
            trades_df["status"] == "CLOSED"
        ]

        open_trades = int(
            (
                trades_df["status"]
                == "OPEN_AT_END"
            ).sum()
        )

    completed_trades = len(
        completed
    )

    if completed_trades > 0:

        avg_trade = completed[
            "trade_return"
        ].mean()

        median_trade = completed[
            "trade_return"
        ].median()

        worst_trade = completed[
            "trade_return"
        ].min()

        best_trade = completed[
            "trade_return"
        ].max()

    else:

        avg_trade = np.nan
        median_trade = np.nan
        worst_trade = np.nan
        best_trade = np.nan

    return {
        "Strategy": strategy,
        "Final Value": final_value,
        "Total Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
        "Sharpe": sharpe,
        "Completed Trades":
            completed_trades,
        "Open Trades":
            open_trades,
        "Avg Trade Return":
            avg_trade,
        "Median Trade Return":
            median_trade,
        "Worst Trade":
            worst_trade,
        "Best Trade":
            best_trade,
    }


# =============================================================================
# ANNUAL RETURNS
# =============================================================================

def calculate_annual_returns(
    equity_df,
    strategy
):

    data = equity_df.copy()

    data["year"] = (
        data["date"].dt.year
    )

    year_end = (
        data
        .groupby("year")["equity"]
        .last()
    )

    annual = {}

    previous_equity = (
        INITIAL_CAPITAL
    )

    for year, value in year_end.items():

        annual[year] = (
            value
            / previous_equity
            - 1.0
        )

        previous_equity = value

    return {
        "Strategy": strategy,
        **annual,
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_strategy(
    df,
    equity_df,
    trades_df,
    strategy
):

    print()
    print(
        f"VALIDATION — {strategy}"
    )
    print("-" * 80)

    # -------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------

    if len(equity_df) != len(df):

        raise ValueError(
            f"Equity rows "
            f"{len(equity_df)} != "
            f"input rows {len(df)}"
        )

    # -------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------

    if equity_df["date"].duplicated().any():

        raise ValueError(
            "Duplicate equity dates"
        )

    # -------------------------------------------------------------
    # Equity
    # -------------------------------------------------------------

    if equity_df["equity"].isna().any():

        raise ValueError(
            "NaN equity detected"
        )

    if (
        equity_df["equity"] <= 0
    ).any():

        raise ValueError(
            "Non-positive equity"
        )

    # -------------------------------------------------------------
    # Returns
    # -------------------------------------------------------------

    if not np.isfinite(
        equity_df["daily_return"]
    ).all():

        raise ValueError(
            "Invalid daily return"
        )

    # -------------------------------------------------------------
    # Position
    # -------------------------------------------------------------

    if not equity_df[
        "position"
    ].isin([0, 1]).all():

        raise ValueError(
            "Position must be 0 or 1"
        )

    # -------------------------------------------------------------
    # Trade chronology
    # -------------------------------------------------------------

    if not trades_df.empty:

        closed = trades_df[
            trades_df["status"] == "CLOSED"
        ]

        if not closed.empty:

            if (
                closed["exit_date"]
                <= closed["entry_date"]
            ).any():

                raise ValueError(
                    "Invalid trade chronology"
                )

    # -------------------------------------------------------------
    # Look-ahead audit
    # -------------------------------------------------------------

    invalid_execution = (
        df["execution_date"].notna()
        & (
            df["execution_date"]
            <= df["date"]
        )
    )

    if invalid_execution.any():

        raise ValueError(
            "Execution date <= signal date"
        )

    print(
        f"Equity rows: "
        f"{len(equity_df):,}"
    )

    print(
        f"Final equity: "
        f"${equity_df['equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Minimum equity: "
        f"${equity_df['equity'].min():,.2f}"
    )

    print(
        f"Maximum equity: "
        f"${equity_df['equity'].max():,.2f}"
    )

    print(
        f"Completed trades: "
        f"{(
            trades_df['status'] == 'CLOSED'
        ).sum() if not trades_df.empty else 0}"
    )

    print(
        f"Open trades at end: "
        f"{(
            trades_df['status']
            == 'OPEN_AT_END'
        ).sum() if not trades_df.empty else 0}"
    )

    print(
        "Validation: PASS"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "NG NORMALIZED 1x — "
        "STRICT EVENT-DRIVEN BACKTEST"
    )
    print(
        "Settlement Signal → "
        "T+1 Actual Open → "
        "Settlement Mark"
    )
    print(
        f"{START_DATE.date()} → "
        f"{END_DATE.date()}"
    )
    print("=" * 90)

    df = load_data()

    print(
        f"Input rows: "
        f"{len(df):,}"
    )

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    # -------------------------------------------------------------
    # Global input validation
    # -------------------------------------------------------------

    required = [
        "date",
        "signal_contract",
        "signal_raw_settlement",
        "execution_date",
        "execution_contract",
        "execution_open",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{missing}"
        )

    if df["date"].duplicated().any():

        raise ValueError(
            "Duplicate signal dates"
        )

    print(
        f"Signal date range: "
        f"{df['date'].iloc[0].date()} → "
        f"{df['date'].iloc[-1].date()}"
    )

    strategies = [
        "Buy & Hold",
        "50/200 DMA",
        "Bollinger Mean Reversion",
        "$2 Contrarian",
    ]

    summary_rows = []
    annual_rows = []

    all_trade_rows = []
    all_equity_rows = []

    for strategy in strategies:

        print()
        print("=" * 90)
        print(strategy)
        print("=" * 90)

        (
            equity_df,
            trades_df,
            events
        ) = run_strategy(
            df,
            strategy
        )

        validate_strategy(
            df,
            equity_df,
            trades_df,
            strategy
        )

        metrics = calculate_metrics(
            equity_df,
            trades_df,
            strategy
        )

        annual = (
            calculate_annual_returns(
                equity_df,
                strategy
            )
        )

        summary_rows.append(
            metrics
        )

        annual_rows.append(
            annual
        )

        if not trades_df.empty:

            all_trade_rows.append(
                trades_df
            )

        all_equity_rows.append(
            equity_df
        )

        print()
        print(
            f"Final value: "
            f"${metrics['Final Value']:,.2f}"
        )

        print(
            f"Total return: "
            f"{metrics['Total Return']:.2%}"
        )

        print(
            f"CAGR: "
            f"{metrics['CAGR']:.2%}"
        )

        print(
            f"Max drawdown: "
            f"{metrics['Max Drawdown']:.2%}"
        )

        print(
            f"Sharpe: "
            f"{metrics['Sharpe']:.3f}"
        )

        print(
            f"Completed trades: "
            f"{metrics['Completed Trades']}"
        )

        print(
            f"Open at end: "
            f"{metrics['Open Trades']}"
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    annual_df = pd.DataFrame(
        annual_rows
    )

    equity_df = pd.concat(
        all_equity_rows,
        ignore_index=True
    )

    if all_trade_rows:

        trades_df = pd.concat(
            all_trade_rows,
            ignore_index=True
        )

    else:

        trades_df = pd.DataFrame()

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    annual_df.to_csv(
        ANNUAL_FILE,
        index=False
    )

    trades_df.to_csv(
        TRADES_FILE,
        index=False
    )

    equity_df.to_csv(
        EQUITY_FILE,
        index=False
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    print(
        summary_df[
            [
                "Strategy",
                "Final Value",
                "Total Return",
                "CAGR",
                "Max Drawdown",
                "Sharpe",
                "Completed Trades",
                "Open Trades",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 90)
    print("ANNUAL RETURNS")
    print("=" * 90)

    print(
        annual_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("SAVED")
    print("=" * 90)

    print(SUMMARY_FILE)
    print(ANNUAL_FILE)
    print(TRADES_FILE)
    print(EQUITY_FILE)


if __name__ == "__main__":
    main()