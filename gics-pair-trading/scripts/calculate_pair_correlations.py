from pathlib import Path

import pandas as pd


RETURNS_PATH = Path("data/processed/returns.parquet")
CANDIDATES_PATH = Path("data/processed/pair_candidates_tier1.parquet")
OUTPUT_PATH = Path("data/processed/pair_correlations_tier1.parquet")

WINDOW = 504
MIN_OBSERVATIONS = 400


def calculate_pair_correlation(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> dict:

    pair_returns = returns[[ticker_a, ticker_b]].dropna()

    observations = len(pair_returns)

    result = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "observations": observations,
        "correlation_window": WINDOW,
        "latest_correlation": None,
        "status": "INELIGIBLE",
    }

    if observations < MIN_OBSERVATIONS:
        return result

    rolling_corr = (
        pair_returns[ticker_a]
        .rolling(WINDOW)
        .corr(pair_returns[ticker_b])
    )

    latest = rolling_corr.dropna()

    if latest.empty:
        result["status"] = "INSUFFICIENT_WINDOW"
        return result

    result["latest_correlation"] = float(latest.iloc[-1])
    result["status"] = "ELIGIBLE"

    return result


def main():
    print("=" * 70)
    print("Calculating Tier 1 Pair Correlations")
    print("=" * 70)

    returns = pd.read_parquet(RETURNS_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)

    print(f"Return matrix shape: {returns.shape}")
    print(f"Candidate pairs: {len(candidates)}")
    print(f"Rolling window: {WINDOW}")
    print(f"Minimum observations: {MIN_OBSERVATIONS}")
    print()

    results = []

    for _, candidate in candidates.iterrows():

        result = calculate_pair_correlation(
            returns=returns,
            ticker_a=candidate["ticker_a"],
            ticker_b=candidate["ticker_b"],
        )

        result["tier"] = candidate["tier"]
        result["group"] = candidate["group"]
        result["subcategory"] = candidate["subcategory"]

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
            "correlation_window",
            "latest_correlation",
            "status",
        ]
    ]

    report = report.sort_values(
        "latest_correlation",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 70)
    print("Correlation Summary")
    print("=" * 70)

    print(
        report["status"]
        .value_counts()
        .to_string()
    )

    print()

    eligible = report[
        report["status"] == "ELIGIBLE"
    ].copy()

    print(f"Eligible pairs: {len(eligible)}")

    if not eligible.empty:
        print()
        print("=" * 70)
        print("Top 20 Pairs by Latest 504-Day Correlation")
        print("=" * 70)

        display_columns = [
            "group",
            "subcategory",
            "ticker_a",
            "ticker_b",
            "observations",
            "latest_correlation",
        ]

        print(
            eligible[display_columns]
            .head(20)
            .to_string(index=False)
        )

        print()
        print("=" * 70)
        print("Correlation Distribution")
        print("=" * 70)

        print(
            eligible["latest_correlation"]
            .describe()
            .to_string()
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Pair correlation calculation: PASSED")


if __name__ == "__main__":
    main()