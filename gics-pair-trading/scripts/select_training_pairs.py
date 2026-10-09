from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLOSE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "close_prices.parquet"
)

CANDIDATES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pair_candidates_tier1.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pair_selection_training_tier1.parquet"
)


TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"

CORRELATION_WINDOW = 504
COINTEGRATION_WINDOW = 504
MIN_OBSERVATIONS = 400
CORRELATION_THRESHOLD = 0.80
ADF_THRESHOLD = 0.05


def calculate_pair_statistics(close, ticker_a, ticker_b):
    pair = close[[ticker_a, ticker_b]].dropna()

    if len(pair) < MIN_OBSERVATIONS:
        return None

    # Training-period data only.
    pair = pair.loc[TRAIN_START:TRAIN_END]

    if len(pair) < MIN_OBSERVATIONS:
        return None

    # Use only the most recent training observations.
    pair = pair.tail(COINTEGRATION_WINDOW)

    if len(pair) < MIN_OBSERVATIONS:
        return None

    log_a = pd.Series(
        pd.Series(pair[ticker_a]).apply(lambda x: __import__("math").log(x)),
        index=pair.index,
    )

    log_b = pd.Series(
        pd.Series(pair[ticker_b]).apply(lambda x: __import__("math").log(x)),
        index=pair.index,
    )

    correlation = log_a.corr(log_b)

    if pd.isna(correlation):
        return None

    # OLS:
    # log(A) = intercept + beta * log(B) + residual
    x = pd.DataFrame(
        {
            "const": 1.0,
            "log_b": log_b.values,
        },
        index=pair.index,
    )

    y = log_a.values

    coefficients = (
        __import__("numpy").linalg.lstsq(
            x.values,
            y,
            rcond=None,
        )[0]
    )

    intercept = float(coefficients[0])
    hedge_ratio = float(coefficients[1])

    residual = log_a - intercept - hedge_ratio * log_b

    adf_result = adfuller(
        residual,
        regression="c",
        autolag="AIC",
    )

    adf_statistic = float(adf_result[0])
    adf_pvalue = float(adf_result[1])

    cointegrated = adf_pvalue < ADF_THRESHOLD

    return {
        "observations": len(pair),
        "latest_training_date": pair.index.max(),
        "training_correlation": float(correlation),
        "intercept": intercept,
        "hedge_ratio": hedge_ratio,
        "adf_statistic": adf_statistic,
        "adf_pvalue": adf_pvalue,
        "cointegrated": cointegrated,
    }


def main():
    print("=" * 80)
    print("TRAINING-PERIOD PAIR SELECTION")
    print("=" * 80)

    print(f"Training period: {TRAIN_START} -> {TRAIN_END}")
    print(f"Correlation threshold: >= {CORRELATION_THRESHOLD}")
    print(f"ADF threshold: < {ADF_THRESHOLD}")
    print()

    close = pd.read_parquet(CLOSE_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)

    close.index = pd.to_datetime(close.index)

    print(f"Price matrix: {close.shape}")
    print(f"Candidate pairs: {len(candidates)}")
    print()

    # Restrict candidates by the training-period correlation.
    # We calculate statistics independently for every candidate.
    rows = []

    for i, candidate in enumerate(
        candidates.itertuples(index=False),
        start=1,
    ):
        ticker_a = candidate.ticker_a
        ticker_b = candidate.ticker_b

        result = calculate_pair_statistics(
            close,
            ticker_a,
            ticker_b,
        )

        if result is None:
            continue

        if result["training_correlation"] < CORRELATION_THRESHOLD:
            continue

        row = {
            "tier": candidate.tier,
            "group": candidate.group,
            "subcategory": candidate.subcategory,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            **result,
        }

        rows.append(row)

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        raise RuntimeError(
            "No training-period pairs passed the correlation threshold."
        )

    # Explicit final training selection:
    # correlation >= 0.80 AND ADF p-value < 0.05
    selected = result_df[
        (result_df["training_correlation"] >= CORRELATION_THRESHOLD)
        & (result_df["adf_pvalue"] < ADF_THRESHOLD)
    ].copy()

    selected = selected.sort_values(
        ["adf_pvalue", "training_correlation"],
        ascending=[True, False],
    ).reset_index(drop=True)

    selected["training_rank"] = range(1, len(selected) + 1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(OUTPUT_PATH, index=False)

    print(f"Pairs passing correlation filter: {len(result_df)}")
    print(f"Training-selected pairs: {len(selected)}")
    print()

    if len(selected):
        print("SELECTED TRAINING PAIRS")
        print("-" * 80)

        display_columns = [
            "training_rank",
            "ticker_a",
            "ticker_b",
            "training_correlation",
            "hedge_ratio",
            "adf_statistic",
            "adf_pvalue",
            "observations",
            "latest_training_date",
        ]

        print(
            selected[display_columns].to_string(index=False)
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("TRAINING PAIR SELECTION: PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()