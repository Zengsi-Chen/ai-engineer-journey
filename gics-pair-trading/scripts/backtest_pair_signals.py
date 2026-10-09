from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


PRICES_PATH = Path("data/processed/close_prices.parquet")
COINTEGRATION_PATH = Path(
    "data/processed/pair_cointegration_tier1.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/pair_signal_backtest_tier1.parquet"
)

ESTIMATION_WINDOW = 504
ZSCORE_WINDOW = 252
MIN_OBSERVATIONS = 400

ENTRY_THRESHOLDS = [
    0.75,
    1.0,
    1.5,
    2.0,
]

FORWARD_DAYS = [
    5,
    10,
    20,
    40,
    60,
]


def estimate_hedge_ratio(
    pair_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
):
    """
    Estimate:

        log(A) = intercept + beta * log(B) + residual
    """

    log_prices = np.log(pair_prices)

    y = log_prices[ticker_a].to_numpy()
    x = log_prices[ticker_b].to_numpy()

    X = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    coefficients = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    intercept = float(coefficients[0])
    hedge_ratio = float(coefficients[1])

    return intercept, hedge_ratio


def calculate_zscore(
    pair_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
):
    """
    Calculate the current z-score using only information
    available through the current signal date.
    """

    if len(pair_prices) < ESTIMATION_WINDOW:
        return None

    estimation_prices = pair_prices.tail(
        ESTIMATION_WINDOW
    )

    intercept, hedge_ratio = estimate_hedge_ratio(
        estimation_prices,
        ticker_a,
        ticker_b,
    )

    log_prices = np.log(estimation_prices)

    spread = (
        log_prices[ticker_a]
        - intercept
        - hedge_ratio * log_prices[ticker_b]
    )

    if len(spread) < ZSCORE_WINDOW:
        return None

    recent_spread = spread.tail(
        ZSCORE_WINDOW
    )

    mean = recent_spread.mean()
    std = recent_spread.std(ddof=1)

    if std <= 0 or not np.isfinite(std):
        return None

    current_zscore = (
        spread.iloc[-1] - mean
    ) / std

    return {
        "zscore": float(current_zscore),
        "intercept": intercept,
        "hedge_ratio": hedge_ratio,
        "spread": float(spread.iloc[-1]),
    }


def main():

    print("=" * 70)
    print("Pair Signal Walk-Forward Backtest")
    print("=" * 70)

    prices = pd.read_parquet(
        PRICES_PATH
    )

    prices.index = pd.to_datetime(
        prices.index
    )

    cointegration = pd.read_parquet(
        COINTEGRATION_PATH
    )

    candidates = cointegration[
        cointegration["cointegrated"] == True
    ].copy()

    print(
        f"Price matrix shape: {prices.shape}"
    )

    print(
        f"Cointegrated pairs: {len(candidates)}"
    )

    print(
        f"Estimation window: {ESTIMATION_WINDOW}"
    )

    print(
        f"Z-score window: {ZSCORE_WINDOW}"
    )

    print()

    all_events = []

    for _, candidate in candidates.iterrows():

        ticker_a = candidate["ticker_a"]
        ticker_b = candidate["ticker_b"]

        pair_prices = prices[
            [ticker_a, ticker_b]
        ].dropna()

        if len(pair_prices) < (
            ESTIMATION_WINDOW
            + max(FORWARD_DAYS)
            + 1
        ):
            print(
                f"Skipping {ticker_a}/{ticker_b}: "
                f"insufficient observations"
            )
            continue

        print(
            f"Processing {ticker_a}/{ticker_b} "
            f"({len(pair_prices)} observations)"
        )

        # --------------------------------------------------
        # Walk-forward through historical signal dates.
        # --------------------------------------------------

        last_signal_index = (
            len(pair_prices)
            - max(FORWARD_DAYS)
            - 1
        )

        for i in range(
            ESTIMATION_WINDOW - 1,
            last_signal_index + 1,
        ):

            historical_prices = pair_prices.iloc[
                : i + 1
            ]

            zscore_result = calculate_zscore(
                historical_prices,
                ticker_a,
                ticker_b,
            )

            if zscore_result is None:
                continue

            zscore = zscore_result[
                "zscore"
            ]

            if not np.isfinite(zscore):
                continue

            signal_date = pair_prices.index[i]

            # --------------------------------------------------
            # Long A when A is relatively cheap.
            # Short B is NOT part of the portfolio.
            #
            # This event is interpreted as:
            # "A is cheap relative to B."
            # --------------------------------------------------

            if zscore <= -min(
                ENTRY_THRESHOLDS
            ):
                direction = "LONG_A"

            elif zscore >= min(
                ENTRY_THRESHOLDS
            ):
                direction = "LONG_B"

            else:
                continue

            # --------------------------------------------------
            # Record future relative-price movement.
            #
            # This is NOT a portfolio return yet.
            # It measures whether the relative relationship
            # subsequently moved in the expected direction.
            # --------------------------------------------------

            for forward_days in FORWARD_DAYS:

                future_index = (
                    i + forward_days
                )

                if future_index >= len(
                    pair_prices
                ):
                    continue

                future_date = pair_prices.index[
                    future_index
                ]

                a_return = (
                    pair_prices[ticker_a].iloc[
                        future_index
                    ]
                    / pair_prices[ticker_a].iloc[i]
                    - 1
                )

                b_return = (
                    pair_prices[ticker_b].iloc[
                        future_index
                    ]
                    / pair_prices[ticker_b].iloc[i]
                    - 1
                )

                relative_return = (
                    a_return - b_return
                )

                if direction == "LONG_A":
                    strategy_return = (
                        relative_return
                    )
                else:
                    strategy_return = (
                        -relative_return
                    )

                for threshold in ENTRY_THRESHOLDS:

                    if abs(zscore) < threshold:
                        continue

                    all_events.append(
                        {
                            "signal_date": signal_date,
                            "future_date": future_date,
                            "ticker_a": ticker_a,
                            "ticker_b": ticker_b,
                            "direction": direction,
                            "entry_threshold": threshold,
                            "zscore": zscore,
                            "forward_days": forward_days,
                            "a_return": a_return,
                            "b_return": b_return,
                            "relative_return": relative_return,
                            "strategy_return": strategy_return,
                            "hedge_ratio": zscore_result[
                                "hedge_ratio"
                            ],
                        }
                    )

    events = pd.DataFrame(
        all_events
    )

    if events.empty:
        raise RuntimeError(
            "No signal events generated."
        )

    events = events.sort_values(
        [
            "ticker_a",
            "ticker_b",
            "signal_date",
            "forward_days",
            "entry_threshold",
        ]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------
    # Summary statistics.
    # ------------------------------------------------------

    summary_rows = []

    for (
        ticker_a,
        ticker_b,
        threshold,
        forward_days,
    ), group in events.groupby(
        [
            "ticker_a",
            "ticker_b",
            "entry_threshold",
            "forward_days",
        ]
    ):

        returns = group[
            "strategy_return"
        ]

        summary_rows.append(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "entry_threshold": threshold,
                "forward_days": forward_days,
                "signal_count": len(group),
                "win_rate": float(
                    (returns > 0).mean()
                ),
                "mean_return": float(
                    returns.mean()
                ),
                "median_return": float(
                    returns.median()
                ),
                "std_return": float(
                    returns.std(ddof=1)
                ),
                "best_return": float(
                    returns.max()
                ),
                "worst_return": float(
                    returns.min()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 70)
    print("Signal Backtest Summary")
    print("=" * 70)

    print(
        summary.sort_values(
            [
                "entry_threshold",
                "forward_days",
                "mean_return",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        ).to_string(index=False)
    )

    print()
    print("=" * 70)
    print("Best Configurations")
    print("=" * 70)

    best = summary.sort_values(
        "mean_return",
        ascending=False,
    ).head(20)

    print(
        best.to_string(index=False)
    )

    print()
    print(
        f"Total signal events: {len(events)}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print(
        "Walk-forward signal backtest: PASSED"
    )


if __name__ == "__main__":
    main()