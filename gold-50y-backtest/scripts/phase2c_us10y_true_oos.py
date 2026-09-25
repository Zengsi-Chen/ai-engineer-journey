import pandas as pd
import numpy as np

MASTER = "data/gold_silver_master_1968_2026_macro_sp500.csv"

# ============================================================
# Helpers
# ============================================================

def perf(r):
    r = r.dropna()
    wealth = (1 + r).cumprod()

    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = wealth.iloc[-1] ** (1 / years) - 1

    peak = wealth.cummax()
    dd = wealth / peak - 1
    mdd = dd.min()

    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252)

    downside = r[r < 0].std()
    sortino = (
        r.mean() / downside * np.sqrt(252)
        if downside > 0 else np.nan
    )

    return {
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "$10k": 10000 * wealth.iloc[-1],
    }


def print_perf(name, r):
    p = perf(r)

    print(
        f"{name:<24}"
        f"CAGR {p['CAGR']*100:7.2f}%   "
        f"MDD {p['MDD']*100:8.2f}%   "
        f"Sharpe {p['Sharpe']:6.2f}   "
        f"Sortino {p['Sortino']:6.2f}   "
        f"$10k ${p['$10k']:,.0f}"
    )


def annual_returns(r):
    return (1 + r).groupby(r.index.year).prod() - 1


def concentration(r):
    a = annual_returns(r).sort_values(ascending=False)

    out = {}
    for n in [1, 5, 10]:
        keep = a.iloc[n:]

        if len(keep) == 0:
            out[n] = np.nan
            continue

        years = len(keep)
        wealth = (1 + keep).prod()
        out[n] = wealth ** (1 / years) - 1

    return out


def switches(allocation):
    allocation = allocation.dropna()

    return int((allocation != allocation.shift(1)).sum() - 1)


# ============================================================
# Load LOCKED MASTER
# ============================================================

d = pd.read_csv(MASTER, parse_dates=["date"])
d = d.set_index("date").sort_index()

print("=" * 78)
print("PHASE 2C — TRUE OUT-OF-SAMPLE US10Y ALLOCATION AUDIT")
print("=" * 78)

print(
    f"Period: {d.index.min().date()} -> {d.index.max().date()}"
)
print(f"Rows:   {len(d):,}")
print()

# ============================================================
# Returns
# ============================================================

# Asset returns are calculated first.
d["gold_ret"] = d["gold_usd"].pct_change()
d["sp_ret"] = d["sp500_price"].pct_change()
d["sp_tr_ret"] = d["sp500_total_return"].pct_change()

# ============================================================
# TRUE OOS SIGNAL
#
# US10Y(t)
#    ↓
# allocation for t+1
#    ↓
# return(t+1)
#
# Therefore signal is SHIFTED by one observation.
# ============================================================

d["signal_12m"] = d["us10y_12m_change"] > 0
d["signal_3m"] = d["us10y_3m_change"] > 0

# True OOS allocation:
d["alloc_12m"] = d["signal_12m"].shift(1)
d["alloc_3m"] = d["signal_3m"].shift(1)

# ============================================================
# 3M + 12M REGIME SWITCH
#
# A:
#   12M rising -> Gold
#   otherwise  -> S&P
#
# B:
#   both rising -> Gold
#   both falling -> Gold
#   mixed       -> S&P
#
# These reproduce the two Phase 2A definitions,
# but with TRUE OOS timing.
# ============================================================

d["signal_A"] = d["us10y_12m_change"] > 0
d["alloc_A"] = d["signal_A"].shift(1)

d["signal_B"] = (
    (d["us10y_3m_change"] > 0)
    == (d["us10y_12m_change"] > 0)
)

d["alloc_B"] = d["signal_B"].shift(1)

# ============================================================
# Strategy returns
# ============================================================

d["5050_ret"] = (
    0.5 * d["gold_ret"] +
    0.5 * d["sp_ret"]
)

# TRUE 12M
d["switch12_ret"] = np.where(
    d["alloc_12m"],
    d["gold_ret"],
    d["sp_ret"]
)

# TRUE 3M
d["switch3_ret"] = np.where(
    d["alloc_3m"],
    d["gold_ret"],
    d["sp_ret"]
)

# TRUE 3M + 12M A
d["switch_A_ret"] = np.where(
    d["alloc_A"],
    d["gold_ret"],
    d["sp_ret"]
)

# TRUE 3M + 12M B
d["switch_B_ret"] = np.where(
    d["alloc_B"],
    d["gold_ret"],
    d["sp_ret"]
)

# ============================================================
# 1. FULL PERIOD
# ============================================================

print("=" * 78)
print("1. FULL PERIOD PERFORMANCE")
print("=" * 78)

strategies = {
    "Gold B&H": d["gold_ret"],
    "S&P Price B&H": d["sp_ret"],
    "Gold/S&P 50/50": d["5050_ret"],
    "True 12M Switch": d["switch12_ret"],
    "True 3M Switch": d["switch3_ret"],
    "True 3M+12M A": d["switch_A_ret"],
    "True 3M+12M B": d["switch_B_ret"],
}

for name, r in strategies.items():
    print_perf(name, r)

# ============================================================
# 2. SWITCH - 50/50
# ============================================================

print()
print("=" * 78)
print("2. SWITCH - 50/50")
print("=" * 78)

base = perf(d["5050_ret"])

for name in [
    "switch12_ret",
    "switch3_ret",
    "switch_A_ret",
    "switch_B_ret",
]:

    p = perf(d[name])

    print(
        f"{name:<18}"
        f"CAGR Δ {((p['CAGR']-base['CAGR'])*100):+7.2f} pp   "
        f"MDD Δ {((p['MDD']-base['MDD'])*100):+7.2f} pp   "
        f"Sharpe Δ {(p['Sharpe']-base['Sharpe']):+6.2f}"
    )

# ============================================================
# 3. TRADE / REBALANCE COUNT
# ============================================================

print()
print("=" * 78)
print("3. REBALANCE / SWITCH COUNT")
print("=" * 78)

allocations = {
    "True 12M": d["alloc_12m"],
    "True 3M": d["alloc_3m"],
    "True 3M+12M A": d["alloc_A"],
    "True 3M+12M B": d["alloc_B"],
}

for name, a in allocations.items():
    print(f"{name:<22}: {switches(a):,}")

# ============================================================
# 4. ANNUAL RETURNS
# ============================================================

print()
print("=" * 78)
print("4. ANNUAL RETURNS")
print("=" * 78)

annual = pd.DataFrame({
    "Gold": annual_returns(d["gold_ret"]),
    "S&P": annual_returns(d["sp_ret"]),
    "50/50": annual_returns(d["5050_ret"]),
    "12M": annual_returns(d["switch12_ret"]),
    "3M": annual_returns(d["switch3_ret"]),
    "A": annual_returns(d["switch_A_ret"]),
    "B": annual_returns(d["switch_B_ret"]),
})

print(
    (annual * 100).round(2).to_string()
)

# ============================================================
# 5. ANNUAL WIN / LOSS
# ============================================================

print()
print("=" * 78)
print("5. ANNUAL WIN / LOSS")
print("=" * 78)

for col in ["12M", "3M", "A", "B"]:
    x = annual[col].dropna()

    wins = (x > annual["50/50"].reindex(x.index)).sum()
    losses = (x < annual["50/50"].reindex(x.index)).sum()

    print(
        f"{col:<6}: "
        f"beat 50/50 = {wins:2d}, "
        f"lost = {losses:2d}, "
        f"total = {len(x):2d}"
    )

# ============================================================
# 6. TOP 1 / 5 / 10 YEAR REMOVAL
# ============================================================

print()
print("=" * 78)
print("6. TOP 1 / 5 / 10 YEAR REMOVAL")
print("=" * 78)

for name, r in strategies.items():

    c = concentration(r)

    print(
        f"{name:<24}"
        f"remove Top1 {c[1]*100:6.2f}%   "
        f"Top5 {c[5]*100:6.2f}%   "
        f"Top10 {c[10]*100:6.2f}%"
    )

# ============================================================
# 7. DECADE ATTRIBUTION
# ============================================================

print()
print("=" * 78)
print("7. DECADE ATTRIBUTION")
print("=" * 78)

for decade in sorted(
    annual.index.to_series().astype(int) // 10 * 10
):

    years = [
        y for y in annual.index
        if (y // 10) * 10 == decade
    ]

    if not years:
        continue

    print()
    print(f"--- {decade}s ---")

    for col in annual.columns:

        x = annual.loc[years, col].dropna()

        if len(x) == 0:
            continue

        cagr = (1 + x).prod() ** (1 / len(x)) - 1

        print(
            f"{col:<8}: "
            f"CAGR {cagr*100:7.2f}%   "
            f"mean {x.mean()*100:7.2f}%"
        )

# ============================================================
# 8. 1988–2026 S&P TOTAL RETURN COMPARISON
# ============================================================

print()
print("=" * 78)
print("8. 1988–2026 S&P TOTAL RETURN AUDIT")
print("=" * 78)

tr = d.loc[
    d["sp_tr_ret"].notna()
    & d["gold_ret"].notna()
    & d["sp_ret"].notna()
].copy()

tr["5050_tr"] = (
    0.5 * tr["gold_ret"] +
    0.5 * tr["sp_tr_ret"]
)

# Same 12M signal, TRUE OOS
tr["alloc12"] = (
    tr["us10y_12m_change"] > 0
).shift(1)

tr["switch12_tr"] = np.where(
    tr["alloc12"],
    tr["gold_ret"],
    tr["sp_tr_ret"]
)

print_perf("Gold", tr["gold_ret"])
print_perf("S&P Total Return", tr["sp_tr_ret"])
print_perf("Gold/S&P TR 50/50", tr["5050_tr"])
print_perf("True 12M + S&P TR", tr["switch12_tr"])

p5050 = perf(tr["5050_tr"])
p12 = perf(tr["switch12_tr"])

print()
print(
    f"True 12M - 50/50:"
    f" CAGR {(p12['CAGR']-p5050['CAGR'])*100:+.2f} pp |"
    f" MDD {(p12['MDD']-p5050['MDD'])*100:+.2f} pp |"
    f" Sharpe {(p12['Sharpe']-p5050['Sharpe']):+.2f}"
)

# ============================================================
# 9. SWITCH DIRECTION
# ============================================================

print()
print("=" * 78)
print("9. TIME SPENT IN GOLD vs S&P")
print("=" * 78)

for name, a in allocations.items():

    x = a.dropna()

    gold_pct = x.mean()
    sp_pct = 1 - gold_pct

    print(
        f"{name:<22}"
        f"Gold {gold_pct*100:6.2f}%   "
        f"S&P {sp_pct*100:6.2f}%"
    )

# ============================================================
# 10. SAVE RESULTS
# ============================================================

annual.to_csv(
    "data/phase2c_true_oos_annual_returns.csv"
)

out = pd.DataFrame({
    name: r
    for name, r in strategies.items()
})

out.to_csv(
    "data/phase2c_true_oos_daily_returns.csv"
)

print()
print("=" * 78)
print("FILES SAVED")
print("=" * 78)

print("data/phase2c_true_oos_annual_returns.csv")
print("data/phase2c_true_oos_daily_returns.csv")

print()
print("=" * 78)
print("PHASE 2C COMPLETE")
print("=" * 78)