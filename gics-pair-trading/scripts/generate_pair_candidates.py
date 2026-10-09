from itertools import combinations
from pathlib import Path

import pandas as pd

from src.gics.universe import flatten_universe, load_universe


OUTPUT_PATH = Path("data/processed/pair_candidates_tier1.parquet")


def generate_tier1_pairs(records):
    """
    Tier 1:
    Same economic subcategory.

    Example:
        gold_miners:
            AEM / NEM
            AEM / GOLD
            NEM / GOLD
            ...

    Pairs are unordered:
        AEM / NEM == NEM / AEM
    """

    groups = {}

    for record in records:
        group = record["group"]
        subcategory = record["subcategory"]

        key = (group, subcategory)

        groups.setdefault(key, []).append(record["ticker"])

    pairs = []

    for (group, subcategory), tickers in groups.items():

        for ticker_a, ticker_b in combinations(sorted(tickers), 2):
            pairs.append(
                {
                    "tier": 1,
                    "group": group,
                    "subcategory": subcategory,
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                }
            )

    return pairs


def main():
    print("=" * 70)
    print("Generating Pair Candidates - Tier 1")
    print("=" * 70)

    config = load_universe()

    records = flatten_universe(config["universe"])

    print(f"Universe tickers: {len(records)}")
    print()

    pairs = generate_tier1_pairs(records)

    candidates = pd.DataFrame(pairs)

    print(f"Tier 1 candidate pairs: {len(candidates)}")
    print()

    print("=" * 70)
    print("Candidates by Economic Subcategory")
    print("=" * 70)

    counts = (
        candidates
        .groupby(["group", "subcategory"])
        .size()
        .reset_index(name="pair_count")
        .sort_values("pair_count", ascending=False)
    )

    print(counts.to_string(index=False))

    print()
    print("=" * 70)
    print("Candidate Preview")
    print("=" * 70)

    print(candidates.to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Tier 1 candidate generation: PASSED")


if __name__ == "__main__":
    main()