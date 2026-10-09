from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRICES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "close_prices.parquet"
)

OPEN_PRICES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "open_prices.parquet"
)

# IMPORTANT:
# This is the frozen training-selected pair universe.
TRAINING_PAIRS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pair_selection_training_tier1.parquet"
)

OUTPUT_EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pair_event_backtest_hedged_tier1.parquet"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pair_event_backtest_hedged_summary_tier1.parquet"
)


# ============================================================
# Backtest configuration
# ============================================================

# Training period ends here.
TRAIN_END = pd.Timestamp("2024-12-31")

# OOS starts here.
OOS_START = pd.Timestamp("2025-01-01")

# Z-score uses the previous 252 observations.
# IMPORTANT:
# The current signal date itself is NOT included in the
# rolling mean/std calculation.
ZSCORE_WINDOW = 252

# Entry thresholds.
THRESHOLDS = [1.0, 1.5, 2.0]

# Holding periods measured in pair-valid trading observations.
HORIZONS = [20, 40, 60]


# ============================================================
# Data loading
# ============================================================

def load_data():
    """
    Load close prices, open prices, and frozen training pairs.

    The training pair file contains:
        ticker_a
        ticker_b
        intercept
        hedge_ratio
        training_rank
        etc.

    The intercept and hedge_ratio are treated as FROZEN.
    """

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    prices = pd.read_parquet(PRICES_PATH)
    open_prices = pd.read_parquet(OPEN_PRICES_PATH)
    training_pairs = pd.read_parquet(TRAINING_PAIRS_PATH)

    prices.index = pd.to_datetime(prices.index)
    open_prices.index = pd.to_datetime(open_prices.index)

    prices = prices.sort_index()
    open_prices = open_prices.sort_index()

    required_pair_columns = [
        "ticker_a",
        "ticker_b",
        "intercept",
        "hedge_ratio",
    ]

    missing_columns = [
        c for c in required_pair_columns
        if c not in training_pairs.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training pair file is missing required columns: "
            f"{missing_columns}"
        )

    training_pairs = training_pairs.copy()

    training_pairs["ticker_a"] = (
        training_pairs["ticker_a"].astype(str).str.upper()
    )

    training_pairs["ticker_b"] = (
        training_pairs["ticker_b"].astype(str).str.upper()
    )

    # Remove duplicate pair definitions if any.
    training_pairs = (
        training_pairs
        .drop_duplicates(subset=["ticker_a", "ticker_b"])
        .reset_index(drop=True)
    )

    print(f"Close price rows: {len(prices)}")
    print(f"Close price columns: {len(prices.columns)}")
    print(f"Open price rows: {len(open_prices)}")
    print(f"Open price columns: {len(open_prices.columns)}")
    print(f"Training-selected pairs: {len(training_pairs)}")

    print()
    print("Frozen training pairs:")

    display_columns = [
        c
        for c in [
            "training_rank",
            "ticker_a",
            "ticker_b",
            "intercept",
            "hedge_ratio",
            "adf_pvalue",
        ]
        if c in training_pairs.columns
    ]

    print(
        training_pairs[display_columns]
        .sort_values(
            "training_rank"
            if "training_rank" in training_pairs.columns
            else ["ticker_a", "ticker_b"]
        )
        .to_string(index=False)
    )

    return prices, open_prices, training_pairs


# ============================================================
# Frozen spread
# ============================================================

def calculate_frozen_spread(
    log_a: pd.Series,
    log_b: pd.Series,
    intercept: float,
    hedge_ratio: float,
) -> pd.Series:
    """
    Calculate spread using FROZEN training-period parameters.

        spread_t = log(A_t) - alpha - beta * log(B_t)

    alpha and beta come exclusively from the training-selection file.

    They are never re-estimated during the backtest.
    """

    return log_a - intercept - hedge_ratio * log_b


# ============================================================
# Historical z-score
# ============================================================

def calculate_historical_zscore(
    pair: pd.DataFrame,
    signal_position: int,
    intercept: float,
    hedge_ratio: float,
):
    """
    Calculate the z-score for one signal date.

    IMPORTANT anti-lookahead rule:

        current spread = spread at signal date T

        rolling mean/std =
            previous ZSCORE_WINDOW observations BEFORE T

    Therefore the signal-day spread is NOT used to calculate
    its own mean/std.

    The alpha/beta are frozen training parameters.
    """

    if signal_position < ZSCORE_WINDOW:
        return np.nan

    history = pair.iloc[
        signal_position - ZSCORE_WINDOW:signal_position
    ].copy()

    current = pair.iloc[signal_position]

    log_history_a = np.log(history["close_a"])
    log_history_b = np.log(history["close_b"])

    spread_history = calculate_frozen_spread(
        log_history_a,
        log_history_b,
        intercept,
        hedge_ratio,
    )

    spread_history = spread_history.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(spread_history) < ZSCORE_WINDOW:
        return np.nan

    mean_spread = spread_history.mean()
    std_spread = spread_history.std(ddof=1)

    if not np.isfinite(std_spread) or std_spread <= 0:
        return np.nan

    current_log_a = np.log(current["close_a"])
    current_log_b = np.log(current["close_b"])

    current_spread = calculate_frozen_spread(
        current_log_a,
        current_log_b,
        intercept,
        hedge_ratio,
    )

    zscore = (
        current_spread - mean_spread
    ) / std_spread

    return float(zscore)


# ============================================================
# Pair preparation
# ============================================================

def prepare_pair(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
):
    """
    Create a pair-specific price dataframe.

    Only dates where both stocks have close AND open prices
    are retained.

    This guarantees that T+1 entry and T+H exit are based on
    pair-valid observations.
    """

    required_close = [ticker_a, ticker_b]
    required_open = [ticker_a, ticker_b]

    missing_close = [
        ticker
        for ticker in required_close
        if ticker not in prices.columns
    ]

    missing_open = [
        ticker
        for ticker in required_open
        if ticker not in open_prices.columns
    ]

    if missing_close:
        raise ValueError(
            f"Missing close prices for {ticker_a}/{ticker_b}: "
            f"{missing_close}"
        )

    if missing_open:
        raise ValueError(
            f"Missing open prices for {ticker_a}/{ticker_b}: "
            f"{missing_open}"
        )

    pair = pd.DataFrame(
        {
            "close_a": prices[ticker_a],
            "close_b": prices[ticker_b],
            "open_a": open_prices[ticker_a],
            "open_b": open_prices[ticker_b],
        }
    )

    pair = pair.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    pair = pair.dropna(
        subset=[
            "close_a",
            "close_b",
            "open_a",
            "open_b",
        ]
    )

    pair = pair.sort_index()

    # Prices must be strictly positive for log prices.
    pair = pair[
        (pair["close_a"] > 0)
        & (pair["close_b"] > 0)
        & (pair["open_a"] > 0)
        & (pair["open_b"] > 0)
    ].copy()

    return pair


# ============================================================
# Signal event detection
# ============================================================

def find_events_for_pair(
    pair: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    intercept: float,
    hedge_ratio: float,
    threshold: float,
):
    """
    Generate independent threshold-crossing events.

    LONG_A:
        z <= -threshold

    LONG_B:
        z >= +threshold

    Once an event starts, additional consecutive signal days
    inside the same side do NOT create duplicate events.

    The event becomes eligible for a new signal only after
    the z-score crosses back through zero.
    """

    events = []

    active_side = None

    for position in range(len(pair)):

        signal_date = pair.index[position]

        zscore = calculate_historical_zscore(
            pair=pair,
            signal_position=position,
            intercept=intercept,
            hedge_ratio=hedge_ratio,
        )

        if not np.isfinite(zscore):
            continue

        current_side = None

        if zscore <= -threshold:
            current_side = "LONG_A"

        elif zscore >= threshold:
            current_side = "LONG_B"

        # ----------------------------------------------------
        # Start a new event only if:
        #
        # 1. there is no active event, OR
        # 2. previous event was closed by crossing zero.
        # ----------------------------------------------------

        if current_side is not None:

            if active_side is None:

                events.append(
                    {
                        "ticker_a": ticker_a,
                        "ticker_b": ticker_b,
                        "signal_date": signal_date,
                        "signal_position": position,
                        "signal_type": current_side,
                        "zscore": zscore,
                        "threshold": threshold,
                        "intercept": intercept,
                        "hedge_ratio": hedge_ratio,
                    }
                )

                active_side = current_side

        else:

            # Reset event state only after the spread returns
            # through the zero line.

            if active_side == "LONG_A" and zscore >= 0:
                active_side = None

            elif active_side == "LONG_B" and zscore <= 0:
                active_side = None

    return events


# ============================================================
# Event returns
# ============================================================

def calculate_event_returns(
    event: dict,
    pair: pd.DataFrame,
    horizon: int,
):
    """
    Calculate execution and return information.

    Execution convention:

        Signal:
            T close

        Entry:
            T+1 open

        Exit:
            T+H close

    H is measured in pair-valid trading observations.
    """

    signal_position = event["signal_position"]

    entry_position = signal_position + 1
    exit_position = signal_position + horizon

    # Not enough future data.
    if exit_position >= len(pair):
        return None

    signal_date = pair.index[signal_position]
    entry_date = pair.index[entry_position]
    exit_date = pair.index[exit_position]

    signal_type = event["signal_type"]

    # --------------------------------------------------------
    # Prices
    # --------------------------------------------------------

    signal_close_a = pair.iloc[signal_position]["close_a"]
    signal_close_b = pair.iloc[signal_position]["close_b"]

    entry_open_a = pair.iloc[entry_position]["open_a"]
    entry_open_b = pair.iloc[entry_position]["open_b"]

    exit_close_a = pair.iloc[exit_position]["close_a"]
    exit_close_b = pair.iloc[exit_position]["close_b"]

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if signal_type == "LONG_A":

        # Long A / short B
        return_a = (
            exit_close_a / entry_open_a
        ) - 1.0

        return_b = (
            exit_close_b / entry_open_b
        ) - 1.0

        raw_pair_return = return_a - return_b

    elif signal_type == "LONG_B":

        # Long B / short A
        return_a = (
            exit_close_a / entry_open_a
        ) - 1.0

        return_b = (
            exit_close_b / entry_open_b
        ) - 1.0

        raw_pair_return = return_b - return_a

    else:
        raise ValueError(
            f"Unknown signal type: {signal_type}"
        )

    # --------------------------------------------------------
    # Log-spread movement
    # --------------------------------------------------------

    alpha = event["intercept"]
    beta = event["hedge_ratio"]

    entry_spread = (
        np.log(entry_open_a)
        - alpha
        - beta * np.log(entry_open_b)
    )

    exit_spread = (
        np.log(exit_close_a)
        - alpha
        - beta * np.log(exit_close_b)
    )

    spread_change = exit_spread - entry_spread

    if signal_type == "LONG_A":
        spread_return = spread_change

    else:
        spread_return = -spread_change

    # --------------------------------------------------------
    # Period classification
    # --------------------------------------------------------

    if signal_date <= TRAIN_END:
        sample_type = "TRAIN"

    elif signal_date >= OOS_START:
        sample_type = "OOS"

    else:
        sample_type = "OTHER"

    return {
        "ticker_a": event["ticker_a"],
        "ticker_b": event["ticker_b"],
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,

        "signal_type": signal_type,

        "threshold": event["threshold"],
        "horizon": horizon,

        "zscore": event["zscore"],

        "intercept": alpha,
        "hedge_ratio": beta,

        "signal_close_a": signal_close_a,
        "signal_close_b": signal_close_b,

        "entry_open_a": entry_open_a,
        "entry_open_b": entry_open_b,

        "exit_close_a": exit_close_a,
        "exit_close_b": exit_close_b,

        "return_a": return_a,
        "return_b": return_b,

        "raw_pair_return": raw_pair_return,

        "entry_spread": entry_spread,
        "exit_spread": exit_spread,
        "spread_change": spread_change,
        "spread_return": spread_return,

        "sample_type": sample_type,

        "training_pair_frozen": True,
        "beta_frozen": True,
        "alpha_frozen": True,
    }


# ============================================================
# Summary
# ============================================================

def summarize_results(results: pd.DataFrame):
    """
    Generate performance summary by:

        sample_type
        threshold
        horizon
    """

    if results.empty:
        return pd.DataFrame()

    summary_rows = []

    group_columns = [
        "sample_type",
        "threshold",
        "horizon",
    ]

    for keys, group in results.groupby(group_columns):

        sample_type, threshold, horizon = keys

        returns = group["raw_pair_return"].dropna()

        if len(returns) == 0:
            continue

        positive = returns[returns > 0]

        negative = returns[returns < 0]

        win_rate = (
            (returns > 0).mean()
        )

        mean_return = returns.mean()

        median_return = returns.median()

        std_return = returns.std(ddof=1)

        total_return = returns.sum()

        avg_win = (
            positive.mean()
            if len(positive) > 0
            else np.nan
        )

        avg_loss = (
            negative.mean()
            if len(negative) > 0
            else np.nan
        )

        if avg_loss != 0 and np.isfinite(avg_loss):
            profit_factor = (
                positive.sum()
                / abs(negative.sum())
                if len(negative) > 0
                else np.inf
            )
        else:
            profit_factor = np.inf

        summary_rows.append(
            {
                "sample_type": sample_type,
                "threshold": threshold,
                "horizon": horizon,

                "events": len(group),

                "win_rate": win_rate,

                "mean_return": mean_return,
                "median_return": median_return,
                "std_return": std_return,

                "total_return": total_return,

                "avg_win": avg_win,
                "avg_loss": avg_loss,

                "profit_factor": profit_factor,

                "mean_spread_return":
                    group["spread_return"].mean(),

                "median_spread_return":
                    group["spread_return"].median(),
            }
        )

    return pd.DataFrame(summary_rows)


# ============================================================
# Main backtest
# ============================================================

def main():

    print()
    print("=" * 80)
    print("GICS PAIR EVENT BACKTEST")
    print("=" * 80)

    print()
    print("MODEL:")
    print("31 frozen training-selected pairs")
    print("Frozen training intercept / hedge ratio")
    print("Rolling 252-observation z-score")
    print("Signal at T close")
    print("Entry at T+1 open")
    print("Exit at T+H close")

    print()
    print("CONFIG:")
    print(f"TRAIN_END       = {TRAIN_END.date()}")
    print(f"OOS_START       = {OOS_START.date()}")
    print(f"ZSCORE_WINDOW   = {ZSCORE_WINDOW}")
    print(f"THRESHOLDS      = {THRESHOLDS}")
    print(f"HORIZONS        = {HORIZONS}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    prices, open_prices, training_pairs = load_data()

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    all_results = []

    total_pairs = len(training_pairs)

    for pair_number, row in training_pairs.iterrows():

        ticker_a = row["ticker_a"]
        ticker_b = row["ticker_b"]

        intercept = float(row["intercept"])
        hedge_ratio = float(row["hedge_ratio"])

        training_rank = (
            int(row["training_rank"])
            if "training_rank" in row.index
            else pair_number + 1
        )

        print()
        print("-" * 80)
        print(
            f"[{pair_number + 1}/{total_pairs}] "
            f"{ticker_a}/{ticker_b}"
        )
        print(
            f"Training rank : {training_rank}"
        )
        print(
            f"Frozen alpha  : {intercept:.6f}"
        )
        print(
            f"Frozen beta   : {hedge_ratio:.6f}"
        )

        pair = prepare_pair(
            prices=prices,
            open_prices=open_prices,
            ticker_a=ticker_a,
            ticker_b=ticker_b,
        )

        print(
            f"Pair-valid observations: {len(pair)}"
        )

        if len(pair) <= ZSCORE_WINDOW:
            print(
                "SKIP: insufficient observations "
                "for z-score calculation."
            )
            continue

        for threshold in THRESHOLDS:

            events = find_events_for_pair(
                pair=pair,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                intercept=intercept,
                hedge_ratio=hedge_ratio,
                threshold=threshold,
            )

            print(
                f"  Threshold {threshold:.1f}: "
                f"{len(events)} signal events"
            )

            for event in events:

                for horizon in HORIZONS:

                    result = calculate_event_returns(
                        event=event,
                        pair=pair,
                        horizon=horizon,
                    )

                    if result is None:
                        continue

                    result["training_rank"] = training_rank

                    if "group" in row.index:
                        result["group"] = row["group"]

                    if "subcategory" in row.index:
                        result["subcategory"] = row["subcategory"]

                    if "training_correlation" in row.index:
                        result["training_correlation"] = float(
                            row["training_correlation"]
                        )

                    if "adf_pvalue" in row.index:
                        result["training_adf_pvalue"] = float(
                            row["adf_pvalue"]
                        )

                    all_results.append(result)

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    if not all_results:
        raise RuntimeError(
            "No backtest results were generated."
        )

    results = pd.DataFrame(all_results)

    results = results.sort_values(
        [
            "signal_date",
            "ticker_a",
            "ticker_b",
            "threshold",
            "horizon",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save event results
    # --------------------------------------------------------

    results.to_parquet(
        OUTPUT_EVENTS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize_results(results)

    summary.to_parquet(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    print()
    print(f"Training pairs: {training_pairs.shape[0]}")
    print(f"Backtest rows: {len(results)}")

    print(
        "Unique signal dates:",
        results["signal_date"].nunique(),
    )

    print(
        "Unique pairs:",
        results[
            ["ticker_a", "ticker_b"]
        ].drop_duplicates().shape[0],
    )

    print()

    print("Sample counts:")
    print(
        results["sample_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 80)
    print("OOS PERFORMANCE")
    print("=" * 80)

    oos = summary[
        summary["sample_type"] == "OOS"
    ].copy()

    if oos.empty:

        print("No OOS results available.")

    else:

        display_columns = [
            "threshold",
            "horizon",
            "events",
            "win_rate",
            "mean_return",
            "median_return",
            "total_return",
            "profit_factor",
            "mean_spread_return",
        ]

        print(
            oos[display_columns]
            .sort_values(
                ["threshold", "horizon"]
            )
            .to_string(index=False)
        )

    print()
    print("Output files:")
    print(
        f"Events : {OUTPUT_EVENTS_PATH}"
    )
    print(
        f"Summary: {OUTPUT_SUMMARY_PATH}"
    )

    print()
    print("=" * 80)
    print("BACKTEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
