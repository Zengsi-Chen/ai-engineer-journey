import pandas as pd
import numpy as np

MASTER = "data/gold_silver_master_1968_2026_macro_sp500.csv"

# ============================================================
# Phase 3A — Gold/Silver Ratio Event Study
# Question:
# Does an extreme Gold/Silver Ratio predict future
# Silver relative performance vs Gold?
# ============================================================

d = pd.read_csv(MASTER, parse_dates=["date"])
d = d.set_index("date").sort_index()

# Only rows where both metals are valid
d = d[["gold_usd", "silver_usd", "gold_silver_ratio"]].dropna().copy()

# ------------------------------------------------------------
# 1. GSR distribution
# ------------------------------------------------------------

percentiles = [0.10, 0.20, 0.50, 0.80, 0.90]

print("=" * 78)
print("PHASE 3A — GOLD/SILVER RATIO EVENT STUDY")
print("=" * 78)

print("\nPeriod:")
print(f"{d.index.min().date()} -> {d.index.max().date()}")
print(f"Observations: {len(d):,}")

print("\nGSR Percentiles:")
for p in percentiles:
    print(f"P{int(p*100):>2}: {d['gold_silver_ratio'].quantile(p):.2f}")

print("\nCurrent GSR:")
print(f"{d['gold_silver_ratio'].iloc[-1]:.2f}")

# ------------------------------------------------------------
# 2. Future relative return
# ------------------------------------------------------------

horizons = [21, 63, 126, 252]

for h in horizons:
    d[f"silver_fwd_{h}"] = d["silver_usd"].shift(-h) / d["silver_usd"] - 1
    d[f"gold_fwd_{h}"] = d["gold_usd"].shift(-h) / d["gold_usd"] - 1
    d[f"relative_fwd_{h}"] = (
        d[f"silver_fwd_{h}"] - d[f"gold_fwd_{h}"]
    )

# ------------------------------------------------------------
# 3. Full-sample descriptive regimes
# ------------------------------------------------------------

p10 = d["gold_silver_ratio"].quantile(0.10)
p20 = d["gold_silver_ratio"].quantile(0.20)
p80 = d["gold_silver_ratio"].quantile(0.80)
p90 = d["gold_silver_ratio"].quantile(0.90)

d["regime"] = np.select(
    [
        d["gold_silver_ratio"] <= p10,
        d["gold_silver_ratio"] <= p20,
        d["gold_silver_ratio"] >= p90,
        d["gold_silver_ratio"] >= p80,
    ],
    [
        "Extreme Low",
        "Low",
        "Extreme High",
        "High",
    ],
    default="Normal",
)

print("\n" + "=" * 78)
print("FULL-SAMPLE DESCRIPTIVE EVENT STUDY")
print("(Thresholds use the full sample — NOT a tradable OOS strategy)")
print("=" * 78)

for regime in [
    "Extreme Low",
    "Low",
    "Normal",
    "High",
    "Extreme High",
]:
    x = d[d["regime"] == regime]

    print(f"\n{regime}")
    print(f"Observations: {len(x):,}")

    for h in horizons:
        r = x[f"relative_fwd_{h}"].dropna()

        print(
            f"  {h:>3} obs: "
            f"Mean {r.mean()*100:>7.2f}% | "
            f"Median {r.median()*100:>7.2f}% | "
            f"Win {((r > 0).mean()*100):>6.1f}%"
        )

# ------------------------------------------------------------
# 4. Decade / regime stability
# ------------------------------------------------------------

d["period"] = pd.cut(
    d.index.year,
    bins=[1967, 1979, 1999, 2011, 2019, 2026],
    labels=[
        "1970s",
        "1980s-1990s",
        "2000-2011",
        "2012-2019",
        "2020-2026",
    ],
)

print("\n" + "=" * 78)
print("HIGH GSR — REGIME STABILITY")
print("=" * 78)

high = d[d["regime"] == "High"]

for period, x in high.groupby("period", observed=True):
    print(f"\n{period} | N={len(x):,}")

    for h in horizons:
        r = x[f"relative_fwd_{h}"].dropna()

        if len(r) == 0:
            continue

        print(
            f"  {h:>3}: "
            f"Mean {r.mean()*100:>7.2f}% | "
            f"Median {r.median()*100:>7.2f}% | "
            f"Win {((r > 0).mean()*100):>6.1f}%"
        )

# ------------------------------------------------------------
# 5. Extreme High vs Extreme Low
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("EXTREME HIGH vs EXTREME LOW")
print("=" * 78)

for h in horizons:
    high_r = d.loc[
        d["regime"] == "Extreme High",
        f"relative_fwd_{h}"
    ].dropna()

    low_r = d.loc[
        d["regime"] == "Extreme Low",
        f"relative_fwd_{h}"
    ].dropna()

    print(f"\n{h} observations")
    print(
        f"Extreme High: "
        f"Mean {high_r.mean()*100:.2f}% | "
        f"Median {high_r.median()*100:.2f}% | "
        f"Win {(high_r > 0).mean()*100:.1f}%"
    )
    print(
        f"Extreme Low : "
        f"Mean {low_r.mean()*100:.2f}% | "
        f"Median {low_r.median()*100:.2f}% | "
        f"Win {(low_r > 0).mean()*100:.1f}%"
    )

print("\n" + "=" * 78)
print("PHASE 3A COMPLETE")
print("=" * 78)