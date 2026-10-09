from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


PRICES_PATH = Path("data/processed/close_prices.parquet")
COINTEGRATION_PATH = Path(
    "data/processed/pair_cointegration_tier1.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/pair_spread_tier1.parquet"
)

SPREAD_WINDOW = 504
ZSCORE_WINDOW = 252
MIN_ZSCORE_OBSERVATIONS = 100


def calculate_half_life(spread: pd.Series) -> float | None:
    """
    Estimate mean-reversion half-life from:

        ΔS_t = alpha + lambda * S_(t-1) + error

    Half-life = -ln(2) / lambda

    A negative lambda indicates mean reversion.
    """

    spread = spread.dropna()

    if len(spread) < MIN_ZSCORE_OBSERVATIONS:
        return None

    lagged = spread.shift(1)
    delta = spread.diff()

    regression_data = pd.concat(
        [delta.rename("delta"), lagged.rename("lagged")],
        axis=1,
    ).dropna()

    if len(regression_data) < MIN_ZSCORE_OBSERVATIONS:
        return None

    x = regression_data["lagged"].to_numpy()
    y = regression_data["delta"].to_numpy()

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

    lambda_value = float(coefficients[1])

    if lambda_value >= 0:
        return None

    half_life = -np.log(2) / lambda_value

    if not np.isfinite(half_life):
        return None

    return float(half_life)


def calculate_pair_spread(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    intercept: float,
    hedge_ratio: float,
) -> dict:

    pair_prices = prices[
        [ticker_a, ticker_b]
    ].dropna()

    result = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "observations": len(pair_prices),
        "spread_window": SPREAD_WINDOW,
        "zscore_window": ZSCORE_WINDOW,
        "intercept": intercept,
        "hedge_ratio": hedge_ratio,
        "current_spread": None,
        "spread_mean_252": None,
        "spread_std_252": None,
        "current_zscore": None,
        "half_life": None,
        "status": "INELIGIBLE",
    }

    if len(pair_prices) < SPREAD_WINDOW:
        return result

    # Use the same 504-observation window used by
    # the cointegration regression.
    pair_prices = pair_prices.tail(
        SPREAD_WINDOW
    )

    log_prices = np.log(pair_prices)

    spread = (
        log_prices[ticker_a]
        - intercept
        - hedge_ratio * log_prices[ticker_b]
    )

    # Current spread.
    current_spread = float(
        spread.iloc[-1]
    )

    # 252-observation rolling statistics.
    if len(spread) < ZSCORE_WINDOW:
        return result

    recent_spread = spread.tail(
        ZSCORE_WINDOW
    )

    spread_mean = float(
        recent_spread.mean()
    )

    spread_std = float(
        recent_spread.std(ddof=1)
    )

    if spread_std <= 0 or not np.isfinite(spread_std):
        return result

    current_zscore = (
        current_spread - spread_mean
    ) / spread_std

    half_life = calculate_half_life(
        spread
    )

    result.update(
        {
            "current_spread": current_spread,
            "spread_mean_252": spread_mean,
            "spread_std_252": spread_std,
            "current_zscore": float(
                current_zscore
            ),
            "half_life": half_life,
            "status": "ELIGIBLE",
        }
    )

    return result


def main():

    print("=" * 70)
    print("Tier 1 Pair Spread / Half-Life / Z-Score")
    print("=" * 70)

    prices = pd.read_parquet(
        PRICES_PATH
    )

    cointegration = pd.read_parquet(
        COINTEGRATION_PATH
    )

    candidates = cointegration[
        (
            cointegration["cointegrated"]
            == True
        )
    ].copy()

    print(
        f"Price matrix shape: {prices.shape}"
    )

    print(
        f"Cointegrated pairs: "
        f"{len(candidates)}"
    )

    print(
        f"Spread window: "
        f"{SPREAD_WINDOW}"
    )

    print(
        f"Z-score window: "
        f"{ZSCORE_WINDOW}"
    )

    print()

    results = []

    for _, candidate in candidates.iterrows():

        result = calculate_pair_spread(
            prices=prices,
            ticker_a=candidate[
                "ticker_a"
            ],
            ticker_b=candidate[
                "ticker_b"
            ],
            intercept=float(
                candidate["intercept"]
            ),
            hedge_ratio=float(
                candidate["hedge_ratio"]
            ),
        )

        result["tier"] = candidate[
            "tier"
        ]

        result["group"] = candidate[
            "group"
        ]

        result["subcategory"] = candidate[
            "subcategory"
        ]

        result["latest_correlation"] = float(
            candidate[
                "latest_correlation"
            ]
        )

        result["adf_statistic"] = float(
            candidate[
                "adf_statistic"
            ]
        )

        result["adf_pvalue"] = float(
            candidate[
                "adf_pvalue"
            ]
        )

        results.append(result)

    report = pd.DataFrame(
        results
    )

    report = report[
        [
            "tier",
            "group",
            "subcategory",
            "ticker_a",
            "ticker_b",
            "observations",
            "spread_window",
            "zscore_window",
            "latest_correlation",
            "intercept",
            "hedge_ratio",
            "adf_statistic",
            "adf_pvalue",
            "current_spread",
            "spread_mean_252",
            "spread_std_252",
            "current_zscore",
            "half_life",
            "status",
        ]
    ]

    report = report.sort_values(
        "current_zscore",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(
        drop=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 70)
    print("Spread Analysis")
    print("=" * 70)

    print(
        report["status"]
        .value_counts()
        .to_string()
    )

    print()

    display_columns = [
        "group",
        "subcategory",
        "ticker_a",
        "ticker_b",
        "latest_correlation",
        "adf_pvalue",
        "hedge_ratio",
        "current_zscore",
        "half_life",
    ]

    print(
        report[
            display_columns
        ].to_string(index=False)
    )

    print()

    print("=" * 70)
    print("Most Positive Z-Scores")
    print("=" * 70)

    positive = report.sort_values(
        "current_zscore",
        ascending=False,
    )

    print(
        positive[
            display_columns
        ].head(5).to_string(
            index=False
        )
    )

    print()

    print("=" * 70)
    print("Most Negative Z-Scores")
    print("=" * 70)

    negative = report.sort_values(
        "current_zscore",
        ascending=True,
    )

    print(
        negative[
            display_columns
        ].head(5).to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()

    print(
        "Pair spread calculation: PASSED"
    )


if __name__ == "__main__":
    main()