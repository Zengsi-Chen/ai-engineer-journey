import pandas as pd
import numpy as np

MASTER = "data/gold_silver_master_1968_2026_macro_sp500.csv"


def metrics(r):
    r = r.dropna()
    wealth = (1 + r).cumprod()
    years = len(r) / 252
    cagr = wealth.iloc[-1] ** (1 / years) - 1

    peak = wealth.cummax()
    dd = wealth / peak - 1

    sharpe = np.sqrt(252) * r.mean() / r.std()
    downside = r[r < 0].std()
    sortino = np.sqrt(252) * r.mean() / downside if downside > 0 else np.nan

    return cagr, dd.min(), sharpe, sortino, wealth.iloc[-1]


# ------------------------------------------------------------
# Load locked master
# ------------------------------------------------------------
d = pd.read_csv(MASTER, parse_dates=["date"])
d = d.set_index("date").sort_index()

# Own-period daily returns
d = d[d["tradable_day"]].copy()
d["gold_ret"] = d["gold_usd"].pct_change()
d["sp_ret"] = d["sp500_price"].pct_change()

# ------------------------------------------------------------
# True OOS:
# signal(t) -> allocation(t+1) -> return(t+1)
# ------------------------------------------------------------
d["signal12"] = d["us10y_12m_change"] > 0

d["switch12_ret"] = np.where(
    d["signal12"].shift(1),
    d["gold_ret"],
    d["sp_ret"]
)

d["5050_ret"] = 0.5 * d["gold_ret"] + 0.5 * d["sp_ret"]

d = d.dropna(
    subset=["gold_ret", "sp_ret", "switch12_ret", "5050_ret"]
).copy()


# ============================================================
# 1. Annual attribution
# ============================================================

d["year"] = d.index.year

annual = d.groupby("year").agg(
    gold=("gold_ret", lambda x: (1 + x).prod() - 1),
    sp=("sp_ret", lambda x: (1 + x).prod() - 1),
    fifty_fifty=("5050_ret", lambda x: (1 + x).prod() - 1),
    switch12=("switch12_ret", lambda x: (1 + x).prod() - 1),
)

annual["excess_vs_5050"] = (
    annual["switch12"] - annual["fifty_fifty"]
)

print("\n" + "=" * 78)
print("PHASE 2D — US10Y SIGNAL ATTRIBUTION")
print("=" * 78)

print("\nANNUAL ATTRIBUTION")
print("-" * 78)

print(
    annual[
        [
            "gold",
            "sp",
            "fifty_fifty",
            "switch12",
            "excess_vs_5050",
        ]
    ].to_string(
        float_format=lambda x: f"{x * 100:8.2f}%"
    )
)


# ============================================================
# 2. Top positive / negative contribution years
# ============================================================

print("\n" + "=" * 78)
print("TOP YEARS — SWITCH12 CONTRIBUTION VS 50/50")
print("=" * 78)

print("\nTop 10 positive:")
print(
    annual.sort_values(
        "excess_vs_5050",
        ascending=False
    )[["switch12", "fifty_fifty", "excess_vs_5050"]]
    .head(10)
    .to_string(
        float_format=lambda x: f"{x * 100:8.2f}%"
    )
)

print("\nTop 10 negative:")
print(
    annual.sort_values(
        "excess_vs_5050"
    )[["switch12", "fifty_fifty", "excess_vs_5050"]]
    .head(10)
    .to_string(
        float_format=lambda x: f"{x * 100:8.2f}%"
    )
)


# ============================================================
# 3. Regime attribution
# ============================================================

d["gold_up"] = d["gold_ret"] > 0
d["sp_up"] = d["sp_ret"] > 0

d["asset_regime"] = np.select(
    [
        d["gold_up"] & d["sp_up"],
        d["gold_up"] & ~d["sp_up"],
        ~d["gold_up"] & d["sp_up"],
        ~d["gold_up"] & ~d["sp_up"],
    ],
    [
        "Gold+ / SP+",
        "Gold+ / SP-",
        "Gold- / SP+",
        "Gold- / SP-",
    ],
    default="Unknown",
)

regime = d.groupby("asset_regime").agg(
    observations=("switch12_ret", "size"),
    switch12=("switch12_ret", "mean"),
    fifty_fifty=("5050_ret", "mean"),
)

regime["excess"] = (
    regime["switch12"] - regime["fifty_fifty"]
)

print("\n" + "=" * 78)
print("ASSET REGIME ATTRIBUTION")
print("=" * 78)

print(
    regime.to_string(
        float_format=lambda x: f"{x * 100:8.4f}%"
    )
)


# ============================================================
# 4. Exclusion tests
# ============================================================

def run_period(name, x):
    result = {}

    for label, r in [
        ("50/50", x["5050_ret"]),
        ("12M Switch", x["switch12_ret"]),
    ]:
        cagr, mdd, sharpe, sortino, wealth = metrics(r)

        result[label] = {
            "CAGR": cagr,
            "MDD": mdd,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Final": wealth,
        }

    delta_cagr = (
        result["12M Switch"]["CAGR"]
        - result["50/50"]["CAGR"]
    )

    delta_mdd = (
        result["12M Switch"]["MDD"]
        - result["50/50"]["MDD"]
    )

    print(
        f"{name:<20}"
        f"{result['50/50']['CAGR']*100:8.2f}% "
        f"{result['12M Switch']['CAGR']*100:8.2f}% "
        f"{delta_cagr*100:8.2f}pp "
        f"{result['50/50']['MDD']*100:8.2f}% "
        f"{result['12M Switch']['MDD']*100:8.2f}% "
        f"{delta_mdd*100:8.2f}pp"
    )


print("\n" + "=" * 78)
print("EXCLUSION / SUB-PERIOD ROBUSTNESS")
print("=" * 78)

print(
    f"{'Period':<20}"
    f"{'50/50':>9}"
    f"{'12M':>9}"
    f"{'ΔCAGR':>10}"
    f"{'50/50 MDD':>12}"
    f"{'12M MDD':>10}"
    f"{'ΔMDD':>10}"
)

run_period("Full 1968-2026", d)

run_period(
    "Ex 1970-1982",
    d[~(d.index.year >= 1970) & (d.index.year <= 1982)]
)

run_period(
    "Ex 1970s",
    d[~d.index.year.between(1970, 1979)]
)

run_period(
    "Ex 1970s (1970-79)",
    d[~d.index.to_series().dt.year.between(1970, 1979)]
)

run_period(
    "1983-2026",
    d[d.index.year >= 1983]
)

run_period(
    "1988-2026",
    d[d.index.year >= 1988]
)


# ============================================================
# 5. Signal state attribution
# ============================================================

d["signal_state"] = np.where(
    d["signal12"],
    "US10Y 12M Rising",
    "US10Y 12M Falling/Flat"
)

signal = d.groupby("signal_state").agg(
    observations=("switch12_ret", "size"),
    gold_return=("gold_ret", "mean"),
    sp_return=("sp_ret", "mean"),
    switch_return=("switch12_ret", "mean"),
    fifty_return=("5050_ret", "mean"),
)

signal["switch_excess"] = (
    signal["switch_return"]
    - signal["fifty_return"]
)

print("\n" + "=" * 78)
print("US10Y SIGNAL STATE ATTRIBUTION")
print("=" * 78)

print(
    signal.to_string(
        float_format=lambda x: f"{x * 100:8.4f}%"
    )
)


# ============================================================
# 6. Decade attribution
# ============================================================

d["decade"] = (d.index.year // 10) * 10

decade = d.groupby("decade").agg(
    gold=("gold_ret", lambda x: (1 + x).prod() - 1),
    sp=("sp_ret", lambda x: (1 + x).prod() - 1),
    fifty_fifty=("5050_ret", lambda x: (1 + x).prod() - 1),
    switch12=("switch12_ret", lambda x: (1 + x).prod() - 1),
)

decade["excess_vs_5050"] = (
    decade["switch12"]
    - decade["fifty_fifty"]
)

print("\n" + "=" * 78)
print("DECADE ATTRIBUTION")
print("=" * 78)

print(
    decade.to_string(
        float_format=lambda x: f"{x * 100:8.2f}%"
    )
)


print("\n" + "=" * 78)
print("PHASE 2D COMPLETE")
print("=" * 78)