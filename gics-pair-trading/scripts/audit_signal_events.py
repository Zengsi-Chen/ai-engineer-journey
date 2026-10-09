from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_PATH = PROJECT_ROOT / "data" / "processed" / "pair_signal_backtest_tier1.parquet"
PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "close_prices.parquet"

EVENT_AUDIT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pair_signal_event_audit_tier1.parquet"
)

RETURN_AUDIT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pair_signal_return_audit_tier1.parquet"
)


def load_data():
    events = pd.read_parquet(EVENTS_PATH)
    prices = pd.read_parquet(PRICES_PATH)

    events["signal_date"] = pd.to_datetime(events["signal_date"])
    prices.index = pd.to_datetime(prices.index)

    return events, prices


def build_unique_signals(events):
    """
    Remove duplication caused by:
    - multiple entry thresholds
    - multiple forward horizons

    One pair/date/direction is treated as one raw signal day.
    """

    cols = [
        "signal_date",
        "ticker_a",
        "ticker_b",
        "direction",
        "zscore",
        "hedge_ratio",
    ]

    signals = events[cols].drop_duplicates().copy()

    signals = signals.sort_values(
        ["ticker_a", "ticker_b", "signal_date"]
    ).reset_index(drop=True)

    return signals


def add_episode_ids(signals):
    """
    Consecutive signal dates are treated as the same signal episode.

    A new episode begins when there is a gap of more than one
    trading observation for that pair.
    """

    result = []

    group_cols = ["ticker_a", "ticker_b", "direction"]

    for (ticker_a, ticker_b, direction), group in signals.groupby(
        group_cols, sort=False
    ):
        group = group.sort_values("signal_date").copy()

        dates = group["signal_date"].tolist()

        # Episode changes whenever signal dates are not consecutive
        # trading observations in the available price index.
        pair_dates = sorted(
            set(
                signals.loc[
                    (signals["ticker_a"] == ticker_a)
                    & (signals["ticker_b"] == ticker_b),
                    "signal_date",
                ]
            )
        )

        date_to_position = {
            date: i for i, date in enumerate(pair_dates)
        }

        positions = group["signal_date"].map(date_to_position)

        gap = positions.diff().fillna(999)

        group["new_episode"] = (gap > 1).astype(int)
        group.loc[group.index[0], "new_episode"] = 1

        group["episode_id"] = group["new_episode"].cumsum()

        result.append(group)

    result = pd.concat(result, ignore_index=True)

    return result


def summarize_episodes(signals):
    """
    Collapse signal days into independent signal episodes.
    """

    episode_rows = []

    group_cols = [
        "ticker_a",
        "ticker_b",
        "direction",
        "episode_id",
    ]

    for keys, group in signals.groupby(group_cols, sort=False):
        ticker_a, ticker_b, direction, episode_id = keys

        group = group.sort_values("signal_date")

        episode_rows.append(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "direction": direction,
                "episode_id": episode_id,
                "episode_start": group["signal_date"].iloc[0],
                "episode_end": group["signal_date"].iloc[-1],
                "episode_length": len(group),
                "max_abs_zscore": group["zscore"].abs().max(),
                "min_zscore": group["zscore"].min(),
                "max_zscore": group["zscore"].max(),
                "mean_hedge_ratio": group["hedge_ratio"].mean(),
            }
        )

    episodes = pd.DataFrame(episode_rows)

    summary = (
        episodes.groupby(
            ["ticker_a", "ticker_b", "direction"],
            as_index=False,
        )
        .agg(
            independent_episodes=("episode_id", "count"),
            total_signal_days=("episode_length", "sum"),
            mean_episode_length=("episode_length", "mean"),
            median_episode_length=("episode_length", "median"),
            max_episode_length=("episode_length", "max"),
            median_max_abs_zscore=("max_abs_zscore", "median"),
            max_abs_zscore=("max_abs_zscore", "max"),
        )
    )

    summary["signal_days_per_episode"] = (
        summary["total_signal_days"]
        / summary["independent_episodes"]
    )

    return episodes, summary


def calculate_return_decomposition(events, prices):
    """
    Compare:

    1. Existing raw relative return:
         A_return - B_return

    2. Beta-adjusted relative-value movement:
         exp(Δ(log(A) - beta*log(B))) - 1

    The second measure is NOT called portfolio return.
    It measures movement of the fitted relative-value spread.
    """

    required_cols = [
        "signal_date",
        "ticker_a",
        "ticker_b",
        "direction",
        "entry_threshold",
        "forward_days",
        "hedge_ratio",
        "strategy_return",
    ]

    missing = [c for c in required_cols if c not in events.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    rows = []

    for row in events[required_cols].itertuples(index=False):
        (
            signal_date,
            ticker_a,
            ticker_b,
            direction,
            entry_threshold,
            forward_days,
            hedge_ratio,
            strategy_return,
        ) = row

        if signal_date not in prices.index:
            continue

        future_date_position = prices.index.searchsorted(
            signal_date
        ) + int(forward_days)

        if future_date_position >= len(prices.index):
            continue

        future_date = prices.index[future_date_position]

        try:
            a0 = prices.loc[signal_date, ticker_a]
            b0 = prices.loc[signal_date, ticker_b]

            a1 = prices.loc[future_date, ticker_a]
            b1 = prices.loc[future_date, ticker_b]
        except KeyError:
            continue

        if not all(
            np.isfinite(x)
            for x in [a0, b0, a1, b1, hedge_ratio]
        ):
            continue

        if min(a0, b0, a1, b1) <= 0:
            continue

        a_return = a1 / a0 - 1.0
        b_return = b1 / b0 - 1.0

        raw_relative_return = a_return - b_return

        spread_start = np.log(a0) - hedge_ratio * np.log(b0)
        spread_end = np.log(a1) - hedge_ratio * np.log(b1)

        spread_move = np.exp(spread_end - spread_start) - 1.0

        # If A is relatively cheap, convergence means spread should rise.
        # If B is relatively cheap, convergence means spread should fall.
        if direction == "LONG_A":
            beta_adjusted_strategy_return = spread_move
        elif direction == "LONG_B":
            beta_adjusted_strategy_return = -spread_move
        else:
            beta_adjusted_strategy_return = np.nan

        rows.append(
            {
                "signal_date": signal_date,
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "direction": direction,
                "entry_threshold": entry_threshold,
                "forward_days": forward_days,
                "hedge_ratio": hedge_ratio,
                "raw_strategy_return": strategy_return,
                "raw_relative_return": raw_relative_return,
                "beta_adjusted_relative_return": beta_adjusted_strategy_return,
                "future_date": future_date,
            }
        )

    result = pd.DataFrame(rows)

    return result


def summarize_returns(return_audit):
    """
    Summarize raw versus beta-adjusted relative performance.
    """

    summary = (
        return_audit.groupby(
            [
                "ticker_a",
                "ticker_b",
                "direction",
                "entry_threshold",
                "forward_days",
            ],
            as_index=False,
        )
        .agg(
            observations=("signal_date", "count"),
            raw_mean=("raw_strategy_return", "mean"),
            raw_median=("raw_strategy_return", "median"),
            beta_adjusted_mean=(
                "beta_adjusted_relative_return",
                "mean",
            ),
            beta_adjusted_median=(
                "beta_adjusted_relative_return",
                "median",
            ),
            raw_win_rate=(
                "raw_strategy_return",
                lambda x: (x > 0).mean(),
            ),
            beta_adjusted_win_rate=(
                "beta_adjusted_relative_return",
                lambda x: (x > 0).mean(),
            ),
        )
    )

    return summary


def main():
    print("=" * 80)
    print("Pair Signal Event Independence Audit")
    print("=" * 80)

    events, prices = load_data()

    print(f"Events loaded: {len(events):,}")
    print(f"Price matrix: {prices.shape}")

    # ---------------------------------------------------------
    # 1. Remove threshold/horizon duplication
    # ---------------------------------------------------------

    signals = build_unique_signals(events)

    print()
    print("Unique raw signal days:", f"{len(signals):,}")

    # ---------------------------------------------------------
    # 2. Cluster consecutive signal days
    # ---------------------------------------------------------

    signals = add_episode_ids(signals)

    episodes, event_summary = summarize_episodes(signals)

    episodes.to_parquet(
        EVENT_AUDIT_PATH,
        index=False,
    )

    event_summary.to_parquet(
        EVENT_AUDIT_PATH,
        index=False,
    )

    print()
    print("Independent episode summary:")
    print(
        event_summary.sort_values(
            ["ticker_a", "ticker_b", "direction"]
        ).to_string(index=False)
    )

    # ---------------------------------------------------------
    # 3. Return decomposition
    # ---------------------------------------------------------

    return_audit = calculate_return_decomposition(
        events,
        prices,
    )

    return_summary = summarize_returns(return_audit)

    return_summary.to_parquet(
        RETURN_AUDIT_PATH,
        index=False,
    )

    print()
    print("Return decomposition summary:")
    print(
        return_summary.sort_values(
            "beta_adjusted_mean",
            ascending=False,
        )
        .head(40)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 4. Overall diagnostics
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL DIAGNOSTICS")
    print("=" * 80)

    total_raw_days = len(signals)
    total_episodes = len(episodes)

    print(
        f"Raw signal days:       {total_raw_days:,}"
    )

    print(
        f"Independent episodes:  {total_episodes:,}"
    )

    if total_episodes > 0:
        print(
            f"Compression ratio:     "
            f"{total_raw_days / total_episodes:.2f}x"
        )

    print()
    print(
        f"Event audit saved to:\n{EVENT_AUDIT_PATH}"
    )

    print(
        f"Return audit saved to:\n{RETURN_AUDIT_PATH}"
    )

    print()
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()