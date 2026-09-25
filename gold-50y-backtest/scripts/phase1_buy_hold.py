import numpy as np
import pandas as pd

INPUT = "data/gold_silver_master_1968_2026.csv"
INITIAL = 10_000

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df[
    df["tradable_day"]
    & df["gold_usd"].notna()
    & df["silver_usd"].notna()
].copy()

df = df.sort_values("date").reset_index(drop=True)

prices = df[["gold_usd", "silver_usd"]]
returns = prices.pct_change().dropna()

years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25

strategies = {
    "Gold": [1.0, 0.0],
    "Silver": [0.0, 1.0],
    "50/50": [0.5, 0.5],
    "60/40": [0.6, 0.4],
    "40/60": [0.4, 0.6],
}

print("=" * 72)
print("PHASE 1 — GOLD / SILVER BUY & HOLD BASELINE")
print("=" * 72)
print(f"Period : {df.date.iloc[0].date()} -> {df.date.iloc[-1].date()}")
print(f"Rows   : {len(df):,}")
print(f"Years  : {years:.2f}")
print()

print(
    f"{'Strategy':<10}"
    f"{'CAGR':>10}"
    f"{'MDD':>10}"
    f"{'Sharpe':>10}"
    f"{'Sortino':>10}"
    f"{'$10k Final':>15}"
)
print("-" * 72)

results = {}

for name, w in strategies.items():
    w = np.array(w)

    daily = returns.values @ w
    equity = np.r_[1.0, np.cumprod(1 + daily)]

    cagr = equity[-1] ** (1 / years) - 1

    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    mdd = drawdown.min()

    sharpe = daily.mean() / daily.std() * np.sqrt(252)

    downside = daily[daily < 0]
    sortino = (
        daily.mean() / downside.std() * np.sqrt(252)
        if len(downside) > 1
        else np.nan
    )

    final_value = INITIAL * equity[-1]

    results[name] = {
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Final": final_value,
    }

    print(
        f"{name:<10}"
        f"{cagr:>9.2%}"
        f"{mdd:>10.2%}"
        f"{sharpe:>10.2f}"
        f"{sortino:>10.2f}"
        f"{final_value:>15,.0f}"
    )

print()
print("Baseline interpretation:")
print("- Gold and Silver are measured using the same valid trading dates.")
print("- Missing prices are NOT forward-filled.")
print("- No timing, leverage, transaction costs, or parameter optimization.")
print("- 50/50, 60/40 and 40/60 are daily rebalanced portfolios.")

