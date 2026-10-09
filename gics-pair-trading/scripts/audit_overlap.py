from pathlib import Path

import pandas as pd


BACKTEST_PATH = Path("data/processed/pair_event_backtest_tier1.parquet")

THRESHOLD = 2.0
HORIZON = 40


def main():
    df = pd.read_parquet(BACKTEST_PATH)

    df = df[
        (df["sample_type"] == "OOS")
        & (df["threshold"] == THRESHOLD)
        & (df["horizon"] == HORIZON)
    ].copy()

    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    df = df.sort_values(
        ["ticker_a", "ticker_b", "entry_date"]
    ).reset_index(drop=True)

    print("=" * 80)
    print("OVERLAP AUDIT")
    print("=" * 80)
    print(f"Strategy: threshold={THRESHOLD}, horizon={HORIZON}")
    print(f"OOS events: {len(df)}")
    print()

    overlap_pairs = set()
    overlapped_events = set()

    # Check overlap only within the same pair.
    for i in range(len(df)):
        current = df.iloc[i]

        for j in range(i):
            previous = df.iloc[j]

            if (
                current["ticker_a"] == previous["ticker_a"]
                and current["ticker_b"] == previous["ticker_b"]
                and previous["exit_date"] >= current["entry_date"]
            ):
                overlap_pairs.add((j, i))
                overlapped_events.add(j)
                overlapped_events.add(i)

    print(f"Overlapping event pairs: {len(overlap_pairs)}")
    print(f"Events involved in overlap: {len(overlapped_events)}")

    if len(df) > 0:
        overlap_rate = len(overlapped_events) / len(df)
    else:
        overlap_rate = 0.0

    print(f"Overlap event rate: {overlap_rate:.4f}")

    # Maximum simultaneous positions within the same pair.
    max_concurrent_same_pair = 0

    pair_groups = df.groupby(["ticker_a", "ticker_b"])

    for pair, group in pair_groups:
        events = group.to_dict("records")

        for event in events:
            entry_date = event["entry_date"]

            concurrent = sum(
                1
                for other in events
                if other["entry_date"] <= entry_date <= other["exit_date"]
            )

            max_concurrent_same_pair = max(
                max_concurrent_same_pair,
                concurrent,
            )

    print(
        "Max concurrent positions within same pair:",
        max_concurrent_same_pair,
    )

    print("=" * 80)


if __name__ == "__main__":
    main()