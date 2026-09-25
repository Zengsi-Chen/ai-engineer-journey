import pandas as pd
import numpy as np

MASTER = "data/gold_silver_master_1968_2026_macro_sp500.csv"
OOS = "data/phase3a5_gsr_rolling_oos.csv"

print("=" * 78)
print("PHASE 3C — GSR CONDITIONAL REGIME AUDIT")
print("=" * 78)

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------
m = pd.read_csv(MASTER, parse_dates=["date"])
m = m.set_index("date").sort_index()

d = pd.read_csv(OOS, parse_dates=["date"])
d = d.set_index("date").sort_index()

# ---------------------------------------------------------
# Align required market data
# ---------------------------------------------------------
d = d.join(
    m[["gold_usd", "silver_usd", "sp500_price", "us10y_12m_change"]],
    how="left",
    rsuffix="_master",
)

# ---------------------------------------------------------
# Long-term trends
# ---------------------------------------------------------
d["gold_ma200"] = m["gold_usd"].rolling(200).mean()
d["silver_ma200"] = m["silver_usd"].rolling(200).mean()

d["gold_above_ma200"] = d["gold_usd"] > d["gold_ma200"]
d["silver_above_ma200"] = d["silver_usd"] > d["silver_ma200"]

# ---------------------------------------------------------
# Silver volatility
# ---------------------------------------------------------
silver_ret = m["silver_usd"].pct_change()

d["silver_vol20"] = silver_ret.rolling(20).std() * np.sqrt(252)

# Expanding historical median, strictly past-only
d["vol_median"] = (
    d["silver_vol20"]
    .expanding(1260)
    .median()
    .shift(1)
)

d["silver_high_vol"] = d["silver_vol20"] > d["vol_median"]

# ---------------------------------------------------------
# Future relative returns
# Silver minus Gold
# ---------------------------------------------------------
for h in [21, 63, 126, 252]:
    d[f"silver_future_{h}"] = (
        d["silver_usd"].shift(-h) / d["silver_usd"] - 1
    )

    d[f"gold_future_{h}"] = (
        d["gold_usd"].shift(-h) / d["gold_usd"] - 1
    )

    d[f"relative_future_{h}"] = (
        d[f"silver_future_{h}"] - d[f"gold_future_{h}"]
    )

# ---------------------------------------------------------
# Rolling OOS Extreme High
# ---------------------------------------------------------
d["extreme_high"] = (
    d["gold_silver_ratio"] >= d["p90"]
)

# ---------------------------------------------------------
# Three pre-defined regimes
# ---------------------------------------------------------
d["R1_Extreme_High"] = d["extreme_high"]

d["R2_Extreme_High_Gold_MA200"] = (
    d["extreme_high"]
    & d["gold_above_ma200"].fillna(False)
)

d["R3_Extreme_High_Gold_MA200_HighVol"] = (
    d["extreme_high"]
    & d["gold_above_ma200"].fillna(False)
    & d["silver_high_vol"].fillna(False)
)

regimes = [
    "R1_Extreme_High",
    "R2_Extreme_High_Gold_MA200",
    "R3_Extreme_High_Gold_MA200_HighVol",
]

# ---------------------------------------------------------
# Summary function
# ---------------------------------------------------------
def summarize(mask, name):
    x = d.loc[mask].copy()

    print("\n" + "-" * 78)
    print(name)
    print("-" * 78)
    print(f"N = {len(x):,}")

    for h in [21, 63, 126, 252]:
        r = x[f"relative_future_{h}"].dropna()

        if len(r) == 0:
            continue

        print(
            f"{h:3d}: "
            f"Mean {r.mean():8.2%} | "
            f"Median {r.median():8.2%} | "
            f"Win {((r > 0).mean()):6.1%}"
        )

# ---------------------------------------------------------
# Overall results
# ---------------------------------------------------------
for regime in regimes:
    summarize(d[regime], regime)

# ---------------------------------------------------------
# Annual analysis
# ---------------------------------------------------------
print("\n" + "=" * 78)
print("ANNUAL REGIME AUDIT")
print("=" * 78)

for regime in regimes:

    print("\n" + "-" * 78)
    print(regime)
    print("-" * 78)

    x = d.loc[d[regime]].copy()
    x["year"] = x.index.year

    for h in [63, 126, 252]:

        annual = (
            x.groupby("year")[f"relative_future_{h}"]
            .agg(["count", "mean"])
        )

        print(f"\n{h}-observation relative return:")
        print(
            annual.to_string(
                formatters={
                    "mean": lambda z: f"{z:.2%}"
                }
            )
        )

# ---------------------------------------------------------
# Regime overlap / concentration
# ---------------------------------------------------------
print("\n" + "=" * 78)
print("REGIME OVERLAP")
print("=" * 78)

for regime in regimes:
    print(f"{regime:42s}: {d[regime].sum():6,d}")

r2_share = (
    d["R2_Extreme_High_Gold_MA200"].sum()
    / d["R1_Extreme_High"].sum()
)

r3_share = (
    d["R3_Extreme_High_Gold_MA200_HighVol"].sum()
    / d["R2_Extreme_High_Gold_MA200"].sum()
)

print("R2 share of R1:", f"{r2_share:.1%}")
print("R3 share of R2:", f"{r3_share:.1%}")

# ---------------------------------------------------------
# Top / bottom concentration
# ---------------------------------------------------------
print("\n" + "=" * 78)
print("TOP / BOTTOM OBSERVATION CONCENTRATION")
print("=" * 78)

for regime in regimes:

    x = d.loc[d[regime], "relative_future_252"].dropna()

    if len(x) == 0:
        continue

    total = x.sum()

    top10 = x.nlargest(min(10, len(x))).sum()
    bottom10 = x.nsmallest(min(10, len(x))).sum()

    print(f"\n{regime}")
    print(f"Total 252d relative-return sum : {total:.2%}")
    print(f"Top 10 contribution             : {top10:.2%}")
    print(f"Bottom 10 contribution          : {bottom10:.2%}")

# ---------------------------------------------------------
# Historical era stability
# ---------------------------------------------------------
print("\n" + "=" * 78)
print("ERA STABILITY")
print("=" * 78)

eras = {
    "1973-1979": (1973, 1979),
    "1980-1999": (1980, 1999),
    "2000-2011": (2000, 2011),
    "2012-2019": (2012, 2019),
    "2020-2026": (2020, 2026),
}

for regime in regimes:

    print("\n" + "-" * 78)
    print(regime)
    print("-" * 78)

    for era, (y1, y2) in eras.items():

        mask = (
            d[regime]
            & (d.index.year >= y1)
            & (d.index.year <= y2)
        )

        x = d.loc[mask, "relative_future_252"].dropna()

        if len(x) == 0:
            print(f"{era:12s}: N=0")
            continue

        print(
            f"{era:12s}: "
            f"N={len(x):4d} | "
            f"Mean={x.mean():7.2%} | "
            f"Median={x.median():7.2%} | "
            f"Win={(x > 0).mean():6.1%}"
        )

print("\n" + "=" * 78)
print("PHASE 3C COMPLETE")
print("=" * 78)