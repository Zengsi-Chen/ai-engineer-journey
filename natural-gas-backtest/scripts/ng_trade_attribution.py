import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

INPUT = (
    DATA_DIR /
    "ng_robustness_threshold_trades.csv"
)

OUTPUT_TRADES = (
    DATA_DIR /
    "ng_trade_attribution.csv"
)

OUTPUT_ANNUAL = (
    DATA_DIR /
    "ng_annual_trade_attribution.csv"
)

OUTPUT_CONCENTRATION = (
    DATA_DIR /
    "ng_profit_concentration.csv"
)

OUTPUT_LEAVEOUT = (
    DATA_DIR /
    "ng_remove_one_trade.csv"
)

INITIAL_CAPITAL = 10_000.0

START_DATE = pd.Timestamp(
    "2021-09-17"
)

END_DATE = pd.Timestamp(
    "2026-09-17"
)


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, columns):

    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )


def safe_mean(series):

    if len(series) == 0:
        return np.nan

    return float(series.mean())


def safe_median(series):

    if len(series) == 0:
        return np.nan

    return float(series.median())


def compound_returns(returns):

    if len(returns) == 0:
        return 0.0

    return float(
        np.prod(
            1.0 + returns
        ) - 1.0
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_trades():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"{INPUT} not found.\n"
            "Run ng_robustness_audit.py first."
        )

    df = pd.read_csv(INPUT)

    required = [

        "entry_threshold",
        "exit_threshold",

        "entry_date",
        "entry_contract",
        "entry_price",

        "exit_signal_date",
        "exit_date",
        "exit_contract",
        "exit_price",

        "holding_days",
        "trade_return",
        "status"
    ]

    require_columns(
        df,
        required
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    for c in [
        "entry_date",
        "exit_signal_date",
        "exit_date"
    ]:

        df[c] = pd.to_datetime(
            df[c],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    for c in [
        "entry_threshold",
        "exit_threshold",
        "entry_price",
        "exit_price",
        "holding_days",
        "trade_return"
    ]:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Only CLOSED trades for performance attribution.
    #
    # OPEN_AT_END trades are preserved separately but are
    # excluded from realized-return statistics.
    # --------------------------------------------------------

    df["is_closed"] = (
        df["status"] == "CLOSED"
    )

    return df


# ============================================================
# TRADE-LEVEL ATTRIBUTION
# ============================================================

def build_trade_attribution(df):

    trades = df.copy()

    closed = trades[
        trades["is_closed"]
    ].copy()

    if closed.empty:

        raise ValueError(
            "No CLOSED trades found."
        )

    # --------------------------------------------------------
    # Trade identifiers
    # --------------------------------------------------------

    closed = closed.sort_values(
        [
            "entry_threshold",
            "exit_threshold",
            "entry_date",
            "exit_date"
        ]
    ).reset_index(drop=True)

    closed.insert(
        0,
        "trade_id",
        np.arange(
            1,
            len(closed) + 1
        )
    )

    # --------------------------------------------------------
    # Calendar fields
    # --------------------------------------------------------

    closed["entry_year"] = (
        closed["entry_date"]
        .dt.year
    )

    closed["exit_year"] = (
        closed["exit_date"]
        .dt.year
    )

    closed["entry_month"] = (
        closed["entry_date"]
        .dt.to_period("M")
        .astype(str)
    )

    closed["exit_month"] = (
        closed["exit_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # --------------------------------------------------------
    # Win / loss
    # --------------------------------------------------------

    closed["win"] = (
        closed["trade_return"] > 0
    )

    closed["loss"] = (
        closed["trade_return"] < 0
    )

    closed["breakeven"] = (
        closed["trade_return"] == 0
    )

    # --------------------------------------------------------
    # Dollar P&L assuming the strategy starts each trade with
    # the current compounded equity.
    #
    # This is NOT the same as simply:
    #
    # INITIAL_CAPITAL * trade_return
    #
    # because the actual backtest compounds capital.
    #
    # We reconstruct the capital path independently for each
    # threshold pair.
    # --------------------------------------------------------

    closed["capital_before"] = np.nan
    closed["capital_after"] = np.nan
    closed["dollar_pnl"] = np.nan

    for (
        entry_level,
        exit_level
    ), group in closed.groupby(
        [
            "entry_threshold",
            "exit_threshold"
        ],
        sort=False
    ):

        capital = INITIAL_CAPITAL

        for idx in group.index:

            r = float(
                closed.loc[
                    idx,
                    "trade_return"
                ]
            )

            pnl = (
                capital * r
            )

            capital_after = (
                capital + pnl
            )

            closed.loc[
                idx,
                "capital_before"
            ] = capital

            closed.loc[
                idx,
                "dollar_pnl"
            ] = pnl

            closed.loc[
                idx,
                "capital_after"
            ] = capital_after

            capital = capital_after

    # --------------------------------------------------------
    # Cumulative return inside each parameter combination
    # --------------------------------------------------------

    closed["cumulative_return"] = np.nan

    for (
        entry_level,
        exit_level
    ), group in closed.groupby(
        [
            "entry_threshold",
            "exit_threshold"
        ],
        sort=False
    ):

        cumulative = (
            (1.0 + group["trade_return"])
            .cumprod()
            - 1.0
        )

        closed.loc[
            group.index,
            "cumulative_return"
        ] = cumulative.values

    return closed


# ============================================================
# PROFIT CONCENTRATION
# ============================================================

def build_profit_concentration(trades):

    rows = []

    for (
        entry_level,
        exit_level
    ), group in trades.groupby(
        [
            "entry_threshold",
            "exit_threshold"
        ],
        sort=True
    ):

        group = group.copy()

        group = group.sort_values(
            "dollar_pnl",
            ascending=False
        ).reset_index(drop=True)

        n = len(group)

        total_pnl = float(
            group["dollar_pnl"].sum()
        )

        total_return = compound_returns(
            group["trade_return"]
        )

        # ----------------------------------------------------
        # Top-N contribution
        # ----------------------------------------------------

        def contribution(n_top):

            if n < n_top:

                return np.nan

            top_pnl = float(
                group
                .head(n_top)["dollar_pnl"]
                .sum()
            )

            if total_pnl == 0:

                return np.nan

            return (
                top_pnl /
                total_pnl
            )

        rows.append({

            "entry_threshold":
                entry_level,

            "exit_threshold":
                exit_level,

            "trades":
                n,

            "total_pnl":
                total_pnl,

            "compound_trade_return":
                total_return,

            "top_1_pnl":
                float(
                    group
                    .head(1)["dollar_pnl"]
                    .sum()
                ),

            "top_2_pnl":
                float(
                    group
                    .head(2)["dollar_pnl"]
                    .sum()
                ),

            "top_3_pnl":
                float(
                    group
                    .head(3)["dollar_pnl"]
                    .sum()
                ),

            "top_1_contribution":
                contribution(1),

            "top_2_contribution":
                contribution(2),

            "top_3_contribution":
                contribution(3),

            "largest_trade_return":
                float(
                    group["trade_return"]
                    .max()
                ),

            "smallest_trade_return":
                float(
                    group["trade_return"]
                    .min()
                ),

            "median_trade_return":
                safe_median(
                    group["trade_return"]
                ),

            "mean_trade_return":
                safe_mean(
                    group["trade_return"]
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# REMOVE-ONE-TRADE ANALYSIS
# ============================================================

def build_leave_one_out(trades):

    rows = []

    for (
        entry_level,
        exit_level
    ), group in trades.groupby(
        [
            "entry_threshold",
            "exit_threshold"
        ],
        sort=True
    ):

        group = group.sort_values(
            "entry_date"
        ).reset_index(
            drop=True
        )

        full_return = compound_returns(
            group["trade_return"]
        )

        full_final_value = (
            INITIAL_CAPITAL *
            (1.0 + full_return)
        )

        n = len(group)

        # ----------------------------------------------------
        # Full strategy
        # ----------------------------------------------------

        rows.append({

            "entry_threshold":
                entry_level,

            "exit_threshold":
                exit_level,

            "removed_trade":
                "NONE",

            "removed_entry_date":
                None,

            "removed_exit_date":
                None,

            "removed_trade_return":
                np.nan,

            "remaining_trades":
                n,

            "compound_return":
                full_return,

            "final_value":
                full_final_value,

            "return_change_vs_full":
                0.0
        })

        # ----------------------------------------------------
        # Remove each individual trade
        # ----------------------------------------------------

        for idx in range(n):

            remaining = group.drop(
                index=idx
            )

            remaining_return = compound_returns(
                remaining["trade_return"]
            )

            remaining_final_value = (
                INITIAL_CAPITAL *
                (1.0 + remaining_return)
            )

            removed_return = float(
                group.loc[
                    idx,
                    "trade_return"
                ]
            )

            rows.append({

                "entry_threshold":
                    entry_level,

                "exit_threshold":
                    exit_level,

                "removed_trade":
                    idx + 1,

                "removed_entry_date":
                    group.loc[
                        idx,
                        "entry_date"
                    ],

                "removed_exit_date":
                    group.loc[
                        idx,
                        "exit_date"
                    ],

                "removed_trade_return":
                    removed_return,

                "remaining_trades":
                    len(remaining),

                "compound_return":
                    remaining_return,

                "final_value":
                    remaining_final_value,

                "return_change_vs_full":
                    (
                        remaining_return -
                        full_return
                    )
            })

    return pd.DataFrame(rows)


# ============================================================
# ANNUAL ATTRIBUTION
# ============================================================

def build_annual_attribution(trades):

    rows = []

    # --------------------------------------------------------
    # Attribute a trade to the year in which it ENTERED.
    #
    # This answers:
    #
    # "Which market years generated the opportunities?"
    #
    # We also preserve exit_year separately.
    # --------------------------------------------------------

    for (
        entry_level,
        exit_level,
        year
    ), group in trades.groupby(
        [
            "entry_threshold",
            "exit_threshold",
            "entry_year"
        ],
        sort=True
    ):

        group = group.copy()

        total_pnl = float(
            group["dollar_pnl"].sum()
        )

        compound_return = compound_returns(
            group["trade_return"]
        )

        rows.append({

            "entry_threshold":
                entry_level,

            "exit_threshold":
                exit_level,

            "entry_year":
                int(year),

            "trades":
                len(group),

            "wins":
                int(group["win"].sum()),

            "losses":
                int(group["loss"].sum()),

            "win_rate":
                (
                    group["win"].mean()
                    if len(group) > 0
                    else np.nan
                ),

            "total_pnl":
                total_pnl,

            "compound_trade_return":
                compound_return,

            "avg_trade_return":
                safe_mean(
                    group["trade_return"]
                ),

            "median_trade_return":
                safe_median(
                    group["trade_return"]
                ),

            "best_trade":
                float(
                    group["trade_return"].max()
                ),

            "worst_trade":
                float(
                    group["trade_return"].min()
                ),

            "avg_holding_days":
                safe_mean(
                    group["holding_days"]
                ),

            "exit_years":
                ",".join(
                    sorted(
                        group["exit_year"]
                        .astype(int)
                        .astype(str)
                        .unique()
                    )
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# DISPLAY
# ============================================================

def display_trade_attribution(df):

    print()
    print("=" * 100)
    print(
        "NG TRADE ATTRIBUTION"
    )
    print("=" * 100)

    display = df.copy()

    for c in [
        "entry_threshold",
        "exit_threshold",
        "entry_price",
        "exit_price",
        "trade_return",
        "dollar_pnl",
        "capital_before",
        "capital_after",
        "cumulative_return"
    ]:

        if c in display.columns:

            display[c] = display[c].map(
                lambda x:
                (
                    f"{x:.2%}"
                    if "return" in c
                    else
                    f"${x:,.2f}"
                    if c in [
                        "entry_price",
                        "exit_price",
                        "dollar_pnl",
                        "capital_before",
                        "capital_after"
                    ]
                    else
                    f"{x:.2f}"
                    if pd.notna(x)
                    else ""
                )
            )

    print(
        display.to_string(
            index=False
        )
    )


def display_concentration(df):

    print()
    print("=" * 100)
    print(
        "NG PROFIT CONCENTRATION"
    )
    print("=" * 100)

    display = df.copy()

    money_cols = [
        "total_pnl",
        "top_1_pnl",
        "top_2_pnl",
        "top_3_pnl"
    ]

    pct_cols = [
        "compound_trade_return",
        "top_1_contribution",
        "top_2_contribution",
        "top_3_contribution",
        "largest_trade_return",
        "smallest_trade_return",
        "median_trade_return",
        "mean_trade_return"
    ]

    for c in money_cols:

        display[c] = display[c].map(
            lambda x:
            f"${x:,.2f}"
            if pd.notna(x)
            else ""
        )

    for c in pct_cols:

        display[c] = display[c].map(
            lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else ""
        )

    print(
        display.to_string(
            index=False
        )
    )


def display_leaveout(df):

    print()
    print("=" * 100)
    print(
        "NG REMOVE-ONE-TRADE ANALYSIS"
    )
    print("=" * 100)

    # Only show actual removals.
    display = df[
        df["removed_trade"] != "NONE"
    ].copy()

    display["removed_trade_return"] = (
        display["removed_trade_return"]
        .map(
            lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else ""
        )
    )

    display["compound_return"] = (
        display["compound_return"]
        .map(
            lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else ""
        )
    )

    display["final_value"] = (
        display["final_value"]
        .map(
            lambda x:
            f"${x:,.2f}"
            if pd.notna(x)
            else ""
        )
    )

    display["return_change_vs_full"] = (
        display["return_change_vs_full"]
        .map(
            lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else ""
        )
    )

    print(
        display.to_string(
            index=False
        )
    )


def display_annual(df):

    print()
    print("=" * 100)
    print(
        "NG ANNUAL TRADE ATTRIBUTION"
    )
    print("=" * 100)

    display = df.copy()

    pct_cols = [
        "win_rate",
        "compound_trade_return",
        "avg_trade_return",
        "median_trade_return",
        "best_trade",
        "worst_trade"
    ]

    for c in pct_cols:

        display[c] = display[c].map(
            lambda x:
            f"{x:.2%}"
            if pd.notna(x)
            else ""
        )

    display["total_pnl"] = (
        display["total_pnl"]
        .map(
            lambda x:
            f"${x:,.2f}"
            if pd.notna(x)
            else ""
        )
    )

    display["avg_holding_days"] = (
        display["avg_holding_days"]
        .map(
            lambda x:
            f"{x:.1f}"
            if pd.notna(x)
            else ""
        )
    )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print(
        "NG STRATEGY TRADE ATTRIBUTION AUDIT"
    )
    print("=" * 100)

    print(
        f"Input: {INPUT}"
    )

    df = load_trades()

    print(
        f"Loaded rows: {len(df)}"
    )

    print(
        f"Closed trades: "
        f"{df['is_closed'].sum()}"
    )

    print(
        f"Open-at-end trades: "
        f"{(~df['is_closed']).sum()}"
    )

    # ========================================================
    # 1. TRADE ATTRIBUTION
    # ========================================================

    trades = build_trade_attribution(
        df
    )

    trades.to_csv(
        OUTPUT_TRADES,
        index=False
    )

    # ========================================================
    # 2. ANNUAL ATTRIBUTION
    # ========================================================

    annual = build_annual_attribution(
        trades
    )

    annual.to_csv(
        OUTPUT_ANNUAL,
        index=False
    )

    # ========================================================
    # 3. PROFIT CONCENTRATION
    # ========================================================

    concentration = (
        build_profit_concentration(
            trades
        )
    )

    concentration.to_csv(
        OUTPUT_CONCENTRATION,
        index=False
    )

    # ========================================================
    # 4. REMOVE-ONE-TRADE
    # ========================================================

    leaveout = (
        build_leave_one_out(
            trades
        )
    )

    leaveout.to_csv(
        OUTPUT_LEAVEOUT,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    display_trade_attribution(
        trades
    )

    display_annual(
        annual
    )

    display_concentration(
        concentration
    )

    display_leaveout(
        leaveout
    )

    # ========================================================
    # FILE SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "SAVED"
    )
    print("=" * 100)

    print(
        f"1. {OUTPUT_TRADES}"
    )

    print(
        f"2. {OUTPUT_ANNUAL}"
    )

    print(
        f"3. {OUTPUT_CONCENTRATION}"
    )

    print(
        f"4. {OUTPUT_LEAVEOUT}"
    )


if __name__ == "__main__":
    main()
