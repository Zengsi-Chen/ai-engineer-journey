from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


RETURNS_PATH = Path("data/processed/returns.parquet")
PRICES_PATH = Path("data/processed/close_prices.parquet")
CORRELATIONS_PATH = Path("data/processed/pair_correlations_tier1.parquet")
OUTPUT_PATH = Path("data/processed/pair_cointegration_tier1.parquet")

CORRELATION_THRESHOLD = 0.80
COINTEGRATION_WINDOW = 504
MIN_OBSERVATIONS = 400
ADF_PVALUE_THRESHOLD = 0.05


def calculate_pair_cointegration(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> dict:

    pair_prices = prices[[ticker_a, ticker_b]].dropna()

    result = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "observations": len(pair_prices),
        "window": COINTEGRATION_WINDOW,
        "intercept": None,
        "hedge_ratio": None,
        "adf_statistic": None,
        "adf_pvalue": None,
        "cointegrated": False,
        "status": "INELIGIBLE",
    }

    if len(pair_prices) < MIN_OBSERVATIONS:
        return result

    # Use the most recent observations only.
    pair_prices = pair_prices.tail(COINTEGRATION_WINDOW)

    log_prices = np.log(pair_prices)

    x = log_prices[ticker_b].to_numpy()
    y = log_prices[ticker_a].to_numpy()

    # OLS:
    #
    # log(A) = alpha + beta * log(B) + residual
    #
    X = np.column_stack([np.ones(len(x)), x])

    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    intercept = float(coefficients[0])
    hedge_ratio = float(coefficients[1])

    residual = y - (intercept + hedge_ratio * x)

    # ADF test on the residual.
    adf_result = adfuller(
        residual,
        regression="c",
        autolag="AIC",
        result_object=True,
    )

    adf_statistic = float(adf_result.statistic)
    adf_pvalue = float(adf_result.pvalue)

    adf_statistic = float(adf_result[0])
    adf_pvalue = float(adf_result[1])

    result.update(
        {
            "intercept": intercept,
            "hedge_ratio": hedge_ratio,
            "adf_statistic": adf_statistic,
            "adf_pvalue": adf_pvalue,
            "cointegrated": adf_pvalue < ADF_PVALUE_THRESHOLD,
            "status": "ELIGIBLE",
        }
    )

    return result


def main():

    print("=" * 70)
    print("Tier 1 Pair Cointegration Test")
    print("=" * 70)

    prices = pd.read_parquet(PRICES_PATH)
    correlations = pd.read_parquet(CORRELATIONS_PATH)

    print(f"Price matrix shape: {prices.shape}")
    print(f"Correlation pairs: {len(correlations)}")
    print(f"Cointegration window: {COINTEGRATION_WINDOW}")
    print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"ADF p-value threshold: {ADF_PVALUE_THRESHOLD}")
    print()

    # Only test pairs with sufficiently high current correlation.
    candidates = correlations[
        (correlations["status"] == "ELIGIBLE")
        & (
            correlations["latest_correlation"]
            >= CORRELATION_THRESHOLD
        )
    ].copy()

    print(
        f"Pairs selected for cointegration testing: "
        f"{len(candidates)}"
    )
    print()

    results = []

    for _, candidate in candidates.iterrows():

        result = calculate_pair_cointegration(
            prices=prices,
            ticker_a=candidate["ticker_a"],
            ticker_b=candidate["ticker_b"],
        )

        result["tier"] = candidate["tier"]
        result["group"] = candidate["group"]
        result["subcategory"] = candidate["subcategory"]
        result["latest_correlation"] = candidate[
            "latest_correlation"
        ]

        results.append(result)

    report = pd.DataFrame(results)

    report = report[
        [
            "tier",
            "group",
            "subcategory",
            "ticker_a",
            "ticker_b",
            "observations",
            "window",
            "latest_correlation",
            "intercept",
            "hedge_ratio",
            "adf_statistic",
            "adf_pvalue",
            "cointegrated",
            "status",
        ]
    ]

    report = report.sort_values(
        [
            "cointegrated",
            "adf_pvalue",
            "latest_correlation",
        ],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 70)
    print("Cointegration Summary")
    print("=" * 70)

    print(
        report["status"]
        .value_counts()
        .to_string()
    )

    print()

    cointegrated = report[
        report["cointegrated"]
    ].copy()

    print(
        f"Cointegrated pairs: "
        f"{len(cointegrated)} / {len(report)}"
    )

    if not cointegrated.empty:

        print()
        print("=" * 70)
        print("Cointegrated Pairs")
        print("=" * 70)

        display_columns = [
            "group",
            "subcategory",
            "ticker_a",
            "ticker_b",
            "latest_correlation",
            "hedge_ratio",
            "adf_statistic",
            "adf_pvalue",
        ]

        print(
            cointegrated[
                display_columns
            ].to_string(index=False)
        )

    print()

    print("=" * 70)
    print("All Tested Pairs")
    print("=" * 70)

    display_columns = [
        "ticker_a",
        "ticker_b",
        "latest_correlation",
        "hedge_ratio",
        "adf_pvalue",
        "cointegrated",
    ]

    print(
        report[
            display_columns
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Cointegration calculation: PASSED")


if __name__ == "__main__":
    main()