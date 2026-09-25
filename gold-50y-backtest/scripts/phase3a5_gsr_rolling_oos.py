import pandas as pd
import numpy as np

MASTER = "data/gold_silver_master_1968_2026_macro_sp500.csv"

# ============================================================
# Phase 3A.5 — Rolling OOS GSR Test
#
# GSR(t) -> percentile based ONLY on past data
#          -> future Silver - Gold relative return
#
# No look-ahead
# No trading strategy
# No parameter optimization
# ============================================================

d = pd.read_csv(MASTER, parse_dates=["date"])
d = d.set_index("date").sort_index()

d = d[["gold_usd", "silver_usd", "gold_silver_ratio"]].dropna().copy()

# ------------------------------------------------------------
# Future relative returns
# ------------------------------------------------------------

horizons = [21, 63, 126, 252]

for h in horizons:
    d[f"relative_fwd_{h}"] = (
        d["silver_usd"].shift(-h) / d["silver_usd"]
        - d["gold_usd"].shift(-h) / d["gold_usd"]
    )

# ------------------------------------------------------------
# Expanding historical percentile
#
# Minimum history = 5 years / 1260 observations
#
# IMPORTANT:
# percentile at t uses data through t-1 only.
# ------------------------------------------------------------

MIN_HISTORY = 1260

past = d["gold_silver_ratio"].shift(1)

d["p10"] = past.expanding(min_periods=MIN_HISTORY).quantile(0.10)
d["p20"] = past.expanding(min_periods=MIN_HISTORY).quantile(0.20)
d["p80"] = past.expanding(min_periods=MIN_HISTORY).quantile(0.80)
d["p90"] = past.expanding(min_periods=MIN_HISTORY).quantile(0.90)

d["regime"] = np.select(
    [
        d["gold_silver_ratio"] <= d["p10"],
        d["gold_silver_ratio"] <= d["p20"],
        d["gold_silver_ratio"] >= d["p90"],
        d["gold_silver_ratio"] >= d["p80"],
    ],
    [
        "Extreme Low",
        "Low",
        "Extreme High",
        "High",
    ],
    default="Normal",
)

# Only observations where historical threshold existed
oos = d[d["p90"].notna()].copy()

print("=" * 78)
print("PHASE 3A.5 — ROLLING OOS GSR TEST")
print("=" * 78)

print("\nOOS starts:")
print(oos.index.min().date())

print(f"OOS observations: {len(oos):,}")

print("\nLatest historical thresholds:")
latest = oos.iloc[-1]

print(f"P10: {latest['p10']:.2f}")
print(f"P20: {latest['p20']:.2f}")
print(f"P80: {latest['p80']:.2f}")
print(f"P90: {latest['p90']:.2f}")
print(f"Current GSR: {latest['gold_silver_ratio']:.2f}")

# ------------------------------------------------------------
# 1. OOS event study
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("ROLLING OOS EVENT STUDY")
print("=" * 78)

for regime in [
    "Extreme Low",
    "Low",
    "Normal",
    "High",
    "Extreme High",
]:

    x = oos[oos["regime"] == regime]

    print(f"\n{regime}")
    print(f"Observations: {len(x):,}")

    for h in horizons:
        r = x[f"relative_fwd_{h}"].dropna()

        if len(r) == 0:
            continue

        print(
            f"  {h:>3}: "
            f"Mean {r.mean()*100:>7.2f}% | "
            f"Median {r.median()*100:>7.2f}% | "
            f"Win {(r > 0).mean()*100:>6.1f}%"
        )

# ------------------------------------------------------------
# 2. Extreme High vs Extreme Low
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("EXTREME HIGH vs EXTREME LOW — ROLLING OOS")
print("=" * 78)

for h in horizons:

    high = oos.loc[
        oos["regime"] == "Extreme High",
        f"relative_fwd_{h}"
    ].dropna()

    low = oos.loc[
        oos["regime"] == "Extreme Low",
        f"relative_fwd_{h}"
    ].dropna()

    print(f"\n{h} observations")

    print(
        f"Extreme High: "
        f"N={len(high):,} | "
        f"Mean {high.mean()*100:.2f}% | "
        f"Median {high.median()*100:.2f}% | "
        f"Win {(high > 0).mean()*100:.1f}%"
    )

    print(
        f"Extreme Low : "
        f"N={len(low):,} | "
        f"Mean {low.mean()*100:.2f}% | "
        f"Median {low.median()*100:.2f}% | "
        f"Win {(low > 0).mean()*100:.1f}%"
    )

# ------------------------------------------------------------
# 3. Historical regime stability
# ------------------------------------------------------------

oos["period"] = pd.cut(
    oos.index.year,
    bins=[1980, 1999, 2011, 2019, 2026],
    labels=[
        "1980s-1990s",
        "2000-2011",
        "2012-2019",
        "2020-2026",
    ],
)

print("\n" + "=" * 78)
print("EXTREME HIGH — REGIME STABILITY")
print("=" * 78)

x = oos[oos["regime"] == "Extreme High"]

for period, group in x.groupby("period", observed=True):

    print(f"\n{period} | N={len(group):,}")

    for h in horizons:

        r = group[f"relative_fwd_{h}"].dropna()

        if len(r) == 0:
            continue

        print(
            f"  {h:>3}: "
            f"Mean {r.mean()*100:>7.2f}% | "
            f"Median {r.median()*100:>7.2f}% | "
            f"Win {(r > 0).mean()*100:>6.1f}%"
        )

# ------------------------------------------------------------
# 4. Save OOS observations for later audit
# ------------------------------------------------------------

output = "data/phase3a5_gsr_rolling_oos.csv"

oos.reset_index().to_csv(output, index=False)

print("\n" + "=" * 78)
print("OUTPUT")
print("=" * 78)

print(output)

print("\n" + "=" * 78)
print("PHASE 3A.5 COMPLETE")
print("=" * 78)