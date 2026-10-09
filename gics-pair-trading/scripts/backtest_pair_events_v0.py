from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRICES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "close_prices.parquet"
)

OPEN_PRICES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "open_prices.parquet"
)

COINT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "pair_cointegration_significant_tier1.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "pair_event_backtest_tier1.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "pair_event_backtest_summary_tier1.parquet"
)

ESTIMATION_WINDOW = 504
ZSCORE_WINDOW = 252

THRESHOLDS = [1.0, 1.5, 2.0]
HORIZONS = [20, 40, 60]

TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")


def load_data():
    prices = pd.read_parquet(PRICES_PATH)
    open_prices = pd.read_parquet(OPEN_PRICES_PATH)
    coint = pd.read_parquet(COINT_PATH)

    prices.index = pd.to_datetime(prices.index)
    open_prices.index = pd.to_datetime(open_prices.index)
    coint["ticker_a"] = coint["ticker_a"].astype(str)
    coint["ticker_b"] = coint["ticker_b"].astype(str)

    return prices, open_prices, coint


def estimate_hedge_ratio(log_a, log_b):
    """
    OLS:

        log(A) = alpha + beta * log(B) + error

    Only historical observations supplied to this function
    are used.
    """

    x = sm.add_constant(log_b)

    model = sm.OLS(log_a, x).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    return float(alpha), float(beta)


def calculate_historical_zscore(
    prices,
    ticker_a,
    ticker_b,
    signal_position,
):
    """
    Calculate beta and z-score using ONLY information available
    before or at signal_position.

    Estimation window:
        previous 504 valid observations

    Z-score:
        previous 252 spread observations
    """

    pair = prices[[ticker_a, ticker_b]].dropna()

    if signal_position < ESTIMATION_WINDOW:
        return None

    if signal_position >= len(pair):
        return None

    estimation = pair.iloc[
        signal_position - ESTIMATION_WINDOW:
        signal_position
    ]

    if len(estimation) < ESTIMATION_WINDOW:
        return None

    log_a = np.log(estimation[ticker_a])
    log_b = np.log(estimation[ticker_b])

    alpha, beta = estimate_hedge_ratio(
        log_a,
        log_b,
    )

    spread_estimation = (
        log_a
        - alpha
        - beta * log_b
    )

    if len(spread_estimation) < ZSCORE_WINDOW:
        return None

    z_window = spread_estimation.iloc[
        -ZSCORE_WINDOW:
    ]

    mean = z_window.mean()
    std = z_window.std(ddof=1)

    if not np.isfinite(std) or std <= 0:
        return None

    current_a = pair.iloc[signal_position][ticker_a]
    current_b = pair.iloc[signal_position][ticker_b]

    current_spread = (
        np.log(current_a)
        - alpha
        - beta * np.log(current_b)
    )

    zscore = (
        current_spread - mean
    ) / std

    return {
        "alpha": alpha,
        "beta": beta,
        "zscore": float(zscore),
        "pair_position": signal_position,
        "signal_date": pair.index[signal_position],
    }


def find_events_for_pair(
    prices,
    ticker_a,
    ticker_b,
):
    """
    Identify independent threshold-crossing events.

    For each threshold:

        LONG_A:
            z <= -threshold

        LONG_B:
            z >= +threshold

    Consecutive signal days belong to the same event.

    A new event requires the z-score to first leave
    the threshold region and later cross it again.
    """

    pair = prices[[ticker_a, ticker_b]].dropna()

    if len(pair) < ESTIMATION_WINDOW + ZSCORE_WINDOW:
        return []

    z_records = []

    for position in range(
        ESTIMATION_WINDOW,
        len(pair),
    ):
        result = calculate_historical_zscore(
            pair,
            ticker_a,
            ticker_b,
            position,
        )

        if result is not None:
            z_records.append(result)

    if not z_records:
        return []

    z_df = pd.DataFrame(z_records)

    events = []

    for threshold in THRESHOLDS:

        for direction in ["LONG_A", "LONG_B"]:

            active = False
            event_number = 0

            for row in z_df.itertuples(index=False):

                z = row.zscore

                if direction == "LONG_A":
                    triggered = z <= -threshold
                else:
                    triggered = z >= threshold

                if not active and triggered:

                    active = True
                    event_number += 1

                    events.append(
                        {
                            "ticker_a": ticker_a,
                            "ticker_b": ticker_b,
                            "direction": direction,
                            "entry_threshold": threshold,
                            "event_number": event_number,
                            "signal_date": row.signal_date,
                            "signal_position": row.pair_position,
                            "zscore": z,
                            "hedge_ratio": row.beta,
                            "alpha": row.alpha,
                        }
                    )

                elif active:

                    if direction == "LONG_A":
                        exited = z > 0
                    else:
                        exited = z < 0

                    if exited:
                        active = False

    return events


def calculate_event_returns(
    event,
    pair,
    open_prices,
):
    """
    Calculate forward relative-value returns.

    Signal is generated at T close.
    Entry occurs at T+1 open.
    Exit occurs at T+H close.

    No look-ahead is introduced by the execution model.
    """

    signal_date = event["signal_date"]

    if signal_date not in pair.index:
        return []

    signal_position = pair.index.get_loc(signal_date)

    results = []

    entry_position = signal_position + 1

    if entry_position >= len(pair):
        return []

    entry_date = pair.index[entry_position]

    a0 = open_prices.loc[
        entry_date,
        event["ticker_a"],
    ]

    b0 = open_prices.loc[
        entry_date,
        event["ticker_b"],
    ]

    if not np.isfinite(a0) or not np.isfinite(b0):
        return []

    beta = event["hedge_ratio"]

    for horizon in HORIZONS:

        future_position = signal_position + horizon

        if future_position >= len(pair):
            continue

        future_date = pair.index[future_position]

        a1 = pair.iloc[future_position][event["ticker_a"]]
        b1 = pair.iloc[future_position][event["ticker_b"]]

        if not all(
            np.isfinite(x)
            for x in [a1, b1]
        ):
            continue

        raw_a_return = a1 / a0 - 1.0
        raw_b_return = b1 / b0 - 1.0

        raw_relative_return = (
            raw_a_return - raw_b_return
        )

        spread_start = (
            np.log(a0)
            - beta * np.log(b0)
        )

        spread_end = (
            np.log(a1)
            - beta * np.log(b1)
        )

        spread_move = (
            np.exp(spread_end - spread_start)
            - 1.0
        )

        if event["direction"] == "LONG_A":
            strategy_return = raw_relative_return
            beta_adjusted_return = spread_move

        else:
            strategy_return = -raw_relative_return
            beta_adjusted_return = -spread_move

        if event["direction"] == "LONG_A":
            correct = beta_adjusted_return > 0
        else:
            correct = beta_adjusted_return > 0

        if event["signal_date"] <= TRAIN_END:
            sample = "TRAIN"
        else:
            sample = "OOS"

        results.append(
            {
                **event,

                "entry_date": entry_date,
                "entry_price_a": a0,
                "entry_price_b": b0,

                "forward_days": horizon,

                "exit_date": future_date,
                "exit_price_a": a1,
                "exit_price_b": b1,

                "raw_a_return": raw_a_return,
                "raw_b_return": raw_b_return,
                "raw_relative_return": raw_relative_return,
                "strategy_return": strategy_return,
                "beta_adjusted_return": beta_adjusted_return,
                "profitable": bool(
                    beta_adjusted_return > 0
                ),
                "sample": sample,
            }
        )

    return results


def summarize(results):
    """
    Aggregate event-level results.

    Note:
    Each event contributes at most one observation
    per threshold/horizon.
    """

    summary = (
        results.groupby(
            [
                "ticker_a",
                "ticker_b",
                "direction",
                "entry_threshold",
                "forward_days",
                "sample",
            ],
            as_index=False,
        )
        .agg(
            events=("event_number", "count"),

            mean_strategy_return=(
                "strategy_return",
                "mean",
            ),

            median_strategy_return=(
                "strategy_return",
                "median",
            ),

            mean_beta_adjusted_return=(
                "beta_adjusted_return",
                "mean",
            ),

            median_beta_adjusted_return=(
                "beta_adjusted_return",
                "median",
            ),

            win_rate=(
                "profitable",
                "mean",
            ),

            max_gain=(
                "beta_adjusted_return",
                "max",
            ),

            max_loss=(
                "beta_adjusted_return",
                "min",
            ),
        )
    )

    return summary


def main():

    print("=" * 80)
    print("EVENT-LEVEL WALK-FORWARD PAIR BACKTEST")
    print("=" * 80)

    prices, open_prices, coint = load_data()

    print(
        f"Price matrix: {prices.shape}"
    )

    pairs = (
        coint[
            ["ticker_a", "ticker_b"]
        ]
        .drop_duplicates()
        .values
        .tolist()
    )

    print(
        f"Pairs under test: {len(pairs)}"
    )

    all_results = []

    for i, (ticker_a, ticker_b) in enumerate(
        pairs,
        start=1,
    ):

        print(
            f"[{i:02d}/{len(pairs):02d}] "
            f"{ticker_a}/{ticker_b}"
        )

        pair = prices[
            [ticker_a, ticker_b]
        ].dropna()

        events = find_events_for_pair(
            prices,
            ticker_a,
            ticker_b,
        )

        print(
            f"    Independent events: "
            f"{len(events):,}"
        )

        for event in events:

            results = calculate_event_returns(
                event,
                pair,
                open_prices,
            )

            all_results.extend(results)

    if not all_results:
        raise RuntimeError(
            "No event-level results generated."
        )

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = summarize(
        results_df
    )

    results_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary_df.to_parquet(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 80)
    print("EVENT-LEVEL SUMMARY")
    print("=" * 80)

    display_columns = [
        "ticker_a",
        "ticker_b",
        "direction",
        "entry_threshold",
        "forward_days",
        "sample",
        "events",
        "mean_beta_adjusted_return",
        "median_beta_adjusted_return",
        "win_rate",
        "max_gain",
        "max_loss",
    ]

    print(
        summary_df
        .sort_values(
            "mean_beta_adjusted_return",
            ascending=False,
        )
        [display_columns]
        .to_string(index=False)
    )

    print()
    print("=" * 80)
    print("OOS RESULTS")
    print("=" * 80)

    oos = summary_df[
        summary_df["sample"] == "OOS"
    ].copy()

    print(
        oos
        .sort_values(
            "mean_beta_adjusted_return",
            ascending=False,
        )
        [display_columns]
        .to_string(index=False)
    )

    print()
    print(
        f"Detailed results saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"Summary saved to:\n"
        f"{SUMMARY_PATH}"
    )

    print()
    print("BACKTEST PASSED")


if __name__ == "__main__":
    main()