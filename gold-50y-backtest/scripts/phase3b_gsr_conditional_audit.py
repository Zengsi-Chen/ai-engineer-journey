import pandas as pd
import numpy as np

MASTER = "data/gold_silver_master_1968_2026_macro_sp500.csv"
OOS = "data/phase3a5_gsr_rolling_oos.csv"

d = pd.read_csv(OOS, parse_dates=["date"]).set_index("date").sort_index()

m = pd.read_csv(MASTER, parse_dates=["date"]).set_index("date").sort_index()

# Add macro variables from locked master
cols = [
    "sp500_price",
    "us10y_12m_change",
]

d = d.join(m[cols], how="left")

# ------------------------------------------------------------
# Trend variables
# ------------------------------------------------------------

d["gold_ma200"] = m["gold_usd"].rolling(200).mean()
d["silver_ma200"] = m["silver_usd"].rolling(200).mean()
d["sp_ma200"] = m["sp500_price"].rolling(200).mean()

d["gold_above_ma200"] = (
    d["gold_usd"] > d["gold_ma200"]
)

d["silver_above_ma200"] = (
    d["silver_usd"] > d["silver_ma200"]
)

d["sp_above_ma200"] = (
    d["sp500_price"] > d["sp_ma200"]
)

# GSR trend: 63 observations
d["gsr_change_63"] = (
    m["gold_silver_ratio"] /
    m["gold_silver_ratio"].shift(63) - 1
)

d["gsr_rising"] = d["gsr_change_63"] > 0

# ------------------------------------------------------------
# Silver volatility
# ------------------------------------------------------------

silver_ret = m["silver_usd"].pct_change()

d["silver_vol20"] = silver_ret.rolling(20).std() * np.sqrt(252)

vol_median = d["silver_vol20"].expanding(1260).median().shift(1)

d["silver_high_vol"] = d["silver_vol20"] > vol_median

# ------------------------------------------------------------
# Extreme High = Rolling OOS P90
# ------------------------------------------------------------

d["extreme_high"] = d["gold_silver_ratio"] >= d["p90"]

# ------------------------------------------------------------
# Conditions
# ------------------------------------------------------------

conditions = {
    "GSR Rising": d["gsr_rising"],
    "GSR Falling": ~d["gsr_rising"],

    "Gold Above MA200": d["gold_above_ma200"],
    "Gold Below MA200": ~d["gold_above_ma200"],

    "Silver Above MA200": d["silver_above_ma200"],
    "Silver Below MA200": ~d["silver_above_ma200"],

    "Silver High Vol": d["silver_high_vol"],
    "Silver Normal Vol": ~d["silver_high_vol"],

    "US10Y Rising": d["us10y_12m_change"] > 0,
    "US10Y Falling": d["us10y_12m_change"] <= 0,

    "S&P Above MA200": d["sp_above_ma200"],
    "S&P Below MA200": ~d["sp_above_ma200"],
}

horizons = [21, 63, 126, 252]

print("=" * 78)
print("PHASE 3B — GSR CONDITIONAL EDGE AUDIT")
print("=" * 78)

print("\nCore sample:")
core = d[d["extreme_high"]].copy()

print(f"Extreme High observations: {len(core):,}")

# ------------------------------------------------------------
# Conditional event study
# ------------------------------------------------------------

for name, condition in conditions.items():

    x = core[condition.loc[core.index]]

    print("\n" + "-" * 78)
    print(name)
    print("-" * 78)

    print(f"N = {len(x):,}")

    for h in horizons:

        col = f"relative_fwd_{h}"
        r = x[col].dropna()

        if len(r) == 0:
            continue

        print(
            f"{h:>3}: "
            f"Mean {r.mean()*100:>7.2f}% | "
            f"Median {r.median()*100:>7.2f}% | "
            f"Win {(r > 0).mean()*100:>6.1f}%"
        )

# ------------------------------------------------------------
# Direct conditional spread
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("CONDITIONAL SPREAD")
print("=" * 78)

pairs = [
    ("GSR Rising", "GSR Falling"),
    ("Gold Above MA200", "Gold Below MA200"),
    ("Silver Above MA200", "Silver Below MA200"),
    ("Silver High Vol", "Silver Normal Vol"),
    ("US10Y Rising", "US10Y Falling"),
    ("S&P Above MA200", "S&P Below MA200"),
]

for a, b in pairs:

    xa = core[conditions[a].loc[core.index]]
    xb = core[conditions[b].loc[core.index]]

    print(f"\n{a} vs {b}")

    for h in horizons:

        ra = xa[f"relative_fwd_{h}"].dropna()
        rb = xb[f"relative_fwd_{h}"].dropna()

        if len(ra) == 0 or len(rb) == 0:
            continue

        spread = ra.mean() - rb.mean()

        print(
            f"{h:>3}: "
            f"{spread*100:+.2f} pp"
        )

print("\n" + "=" * 78)
print("PHASE 3B COMPLETE")
print("=" * 78)