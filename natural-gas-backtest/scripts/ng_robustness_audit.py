import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
INPUT = DATA_DIR / "ng_strategy_execution_2021_2026.csv"

INITIAL_CAPITAL = 10_000.0
START_DATE = "2021-09-17"
END_DATE = "2026-09-17"

ENTRY_LEVELS = [1.50, 1.75, 2.00, 2.25, 2.50]
EXIT_LEVELS = [2.25, 2.50, 2.75, 3.00]


def max_drawdown(equity):
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sharpe(returns):
    r = returns.dropna()

    if len(r) < 2:
        return np.nan

    std = r.std(ddof=1)

    if std == 0:
        return np.nan

    return float(r.mean() / std * np.sqrt(252))


def cagr(final_value, start, end):
    years = (
        pd.Timestamp(end) -
        pd.Timestamp(start)
    ).days / 365.25

    return (
        final_value / INITIAL_CAPITAL
    ) ** (1 / years) - 1


def run(df, entry_level, exit_level):

    equity = INITIAL_CAPITAL

    long = False

    entry_price = np.nan
    entry_date = None
    entry_contract = None

    # ---------------------------------------------------------
    # IMPORTANT:
    # Portfolio's last marked price.
    #
    # This is NOT necessarily df[i-1]["signal_raw_settlement"].
    # After entry, the portfolio is first marked at T+1 settlement.
    # ---------------------------------------------------------
    last_mark = np.nan
    last_mark_date = None

    daily = []
    trades = []

    for i, row in df.iterrows():

        signal_date = row["date"]
        signal_price = float(
            row["signal_raw_settlement"]
        )

        exec_date = row["execution_date"]
        exec_open = row["execution_open"]

        can_execute = (
            pd.notna(exec_date)
            and pd.notna(exec_open)
        )

        ret = 0.0
        event = ""

        # =====================================================
        # FLAT
        # =====================================================

        if not long:

            if (
                signal_price < entry_level
                and can_execute
            ):

                # -------------------------------------------------
                # Signal generated at T settlement.
                # Actual entry happens at T+1 open.
                # -------------------------------------------------

                long = True

                entry_price = float(exec_open)
                entry_date = exec_date
                entry_contract = row["execution_contract"]

                # -------------------------------------------------
                # First portfolio mark:
                #
                # T+1 OPEN -> T+1 SETTLEMENT
                #
                # This is the ONLY return recorded for the
                # entry session.
                # -------------------------------------------------

                mark = row["next_raw_settlement"]

                if pd.notna(mark):

                    mark = float(mark)

                    ret = (
                        mark /
                        entry_price
                        - 1.0
                    )

                    last_mark = mark
                    last_mark_date = exec_date

                event = "BUY"

        # =====================================================
        # LONG
        # =====================================================

        else:

            # =================================================
            # EXIT
            # =================================================

            if (
                signal_price >= exit_level
                and can_execute
            ):

                # -------------------------------------------------
                # We were marked at the previous portfolio mark.
                #
                # Exit happens at today's execution open.
                #
                # Therefore:
                #
                # previous mark -> exit open
                #
                # is the final holding-period return.
                # -------------------------------------------------

                if pd.notna(last_mark):

                    ret = (
                        float(exec_open) /
                        last_mark
                        - 1.0
                    )

                else:

                    ret = 0.0

                trades.append({

                    "entry_date":
                        entry_date,

                    "entry_contract":
                        entry_contract,

                    "entry_price":
                        entry_price,

                    "exit_signal_date":
                        signal_date,

                    "exit_date":
                        exec_date,

                    "exit_contract":
                        row["execution_contract"],

                    "exit_price":
                        float(exec_open),

                    "holding_days":
                        (
                            pd.Timestamp(exec_date)
                            -
                            pd.Timestamp(entry_date)
                        ).days,

                    "trade_return":
                        (
                            float(exec_open) /
                            entry_price
                            - 1.0
                        ),

                    "status":
                        "CLOSED"
                })

                # -------------------------------------------------
                # Position is now flat.
                # -------------------------------------------------

                long = False

                entry_price = np.nan
                entry_date = None
                entry_contract = None

                last_mark = np.nan
                last_mark_date = None

                event = "SELL"

            # =================================================
            # HOLD
            # =================================================

            elif pd.notna(signal_price):

                # -------------------------------------------------
                # After entry, the portfolio is already marked at
                # the previous portfolio mark.
                #
                # Therefore:
                #
                # previous mark -> today's settlement
                #
                # NOT:
                #
                # previous signal settlement -> today's settlement
                # -------------------------------------------------

                if pd.notna(last_mark):

                    ret = (
                        signal_price /
                        last_mark
                        - 1.0
                    )

                else:

                    ret = 0.0

                last_mark = signal_price
                last_mark_date = signal_date

        # =====================================================
        # UPDATE EQUITY
        # =====================================================

        equity *= (1.0 + ret)

        daily.append({

            "date":
                signal_date,

            "equity":
                equity,

            "daily_return":
                ret,

            "event":
                event,

            "position":
                int(long)

        })

    # =========================================================
    # OPEN POSITION AT END
    # =========================================================

    if long:

        final_price = float(
            df.iloc[-1]["signal_raw_settlement"]
        )

        trades.append({

            "entry_date":
                entry_date,

            "entry_contract":
                entry_contract,

            "entry_price":
                entry_price,

            "exit_signal_date":
                None,

            "exit_date":
                None,

            "exit_contract":
                None,

            "exit_price":
                final_price,

            "holding_days":
                (
                    pd.Timestamp(df.iloc[-1]["date"])
                    -
                    pd.Timestamp(entry_date)
                ).days,

            "trade_return":
                final_price /
                entry_price
                - 1.0,

            "status":
                "OPEN_AT_END"
        })

    eq = pd.DataFrame(daily)

    completed = [
        x for x in trades
        if x["status"] == "CLOSED"
    ]

    trade_returns = [
        x["trade_return"]
        for x in completed
    ]

    final_value = float(
        eq["equity"].iloc[-1]
    )

    return {

        "entry":
            entry_level,

        "exit":
            exit_level,

        "final_value":
            final_value,

        "total_return":
            final_value /
            INITIAL_CAPITAL
            - 1,

        "CAGR":
            cagr(
                final_value,
                df["date"].iloc[0],
                df["date"].iloc[-1]
            ),

        "MDD":
            max_drawdown(
                eq["equity"]
            ),

        "Sharpe":
            sharpe(
                eq["daily_return"]
            ),

        "trades":
            len(completed),

        "win_rate":
            (
                np.mean(
                    np.array(trade_returns) > 0
                )
                if trade_returns
                else np.nan
            ),

        "avg_trade":
            (
                np.mean(trade_returns)
                if trade_returns
                else np.nan
            ),

        "median_trade":
            (
                np.median(trade_returns)
                if trade_returns
                else np.nan
            ),

        "best_trade":
            (
                np.max(trade_returns)
                if trade_returns
                else np.nan
            ),

        "worst_trade":
            (
                np.min(trade_returns)
                if trade_returns
                else np.nan
            ),

        "open_at_end":
            int(long),

        "equity":
            eq,

        "trades_detail":
            pd.DataFrame(trades)
    }


def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"{INPUT} not found."
        )

    df = pd.read_csv(INPUT)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["execution_date"] = pd.to_datetime(
        df["execution_date"]
    )

    df = df[
        (df["date"] >= START_DATE)
        &
        (df["date"] <= END_DATE)
    ].copy()

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    required = [

        "date",
        "signal_raw_settlement",
        "execution_date",
        "execution_contract",
        "execution_open",
        "next_raw_settlement"

    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    results = []
    trade_frames = []

    # =========================================================
    # PARAMETER GRID
    # =========================================================

    for entry in ENTRY_LEVELS:

        for exit in EXIT_LEVELS:

            if exit <= entry:
                continue

            r = run(
                df,
                entry,
                exit
            )

            results.append({

                k: v
                for k, v in r.items()
                if k not in
                ("equity", "trades_detail")
            })

            if not r["trades_detail"].empty:

                t = r["trades_detail"].copy()

                t.insert(
                    0,
                    "entry_threshold",
                    entry
                )

                t.insert(
                    1,
                    "exit_threshold",
                    exit
                )

                trade_frames.append(t)

    out = pd.DataFrame(results)

    out = out.sort_values(
        ["entry", "exit"]
    )

    DATA_DIR.mkdir(
        exist_ok=True
    )

    out.to_csv(
        DATA_DIR /
        "ng_robustness_threshold_matrix.csv",
        index=False
    )

    if trade_frames:

        pd.concat(
            trade_frames,
            ignore_index=True
        ).to_csv(
            DATA_DIR /
            "ng_robustness_threshold_trades.csv",
            index=False
        )

    # =========================================================
    # INDIVIDUAL GRIDS
    # =========================================================

    for metric in [
        "CAGR",
        "MDD",
        "Sharpe",
        "total_return",
        "final_value"
    ]:

        out[
            ["entry", "exit", metric]
        ].to_csv(

            DATA_DIR /
            f"ng_robustness_{metric.lower()}_grid.csv",

            index=False
        )

    # =========================================================
    # DISPLAY
    # =========================================================

    print()
    print("=" * 100)
    print(
        "NG STRATEGY ROBUSTNESS AUDIT"
    )
    print(
        "ENTRY / EXIT THRESHOLD SENSITIVITY"
    )
    print("=" * 100)

    print(
        f"Period: "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        "Signal: official CME settlement"
    )

    print(
        "Execution: next settlement "
        "session actual contract open"
    )

    print()

    display = out.copy()

    display["final_value"] = (
        display["final_value"]
        .map(lambda x:
             f"${x:,.2f}")
    )

    for c in [
        "total_return",
        "CAGR",
        "MDD",
        "win_rate",
        "avg_trade",
        "median_trade",
        "best_trade",
        "worst_trade"
    ]:

        display[c] = (
            display[c]
            .map(
                lambda x:
                f"{x:.2%}"
                if pd.notna(x)
                else ""
            )
        )

    display["Sharpe"] = (
        display["Sharpe"]
        .map(
            lambda x:
            f"{x:.3f}"
            if pd.notna(x)
            else ""
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("Saved:")

    print(
        "data\\ng_robustness_threshold_matrix.csv"
    )

    print(
        "data\\ng_robustness_threshold_trades.csv"
    )

    print(
        "data\\ng_robustness_cagr_grid.csv"
    )

    print(
        "data\\ng_robustness_mdd_grid.csv"
    )

    print(
        "data\\ng_robustness_sharpe_grid.csv"
    )


if __name__ == "__main__":
    main()