
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("artifacts/reports/oos_bidirectional_equity.csv")
OUTPUT_DIR = Path("artifacts/reports")

N_BOOTSTRAPS = 5000
BLOCK_LENGTH = 20
TRADING_DAYS_PER_YEAR = 252
RANDOM_SEED = 42


def calculate_metrics(returns):
    returns = np.asarray(returns, dtype=float)

    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    total_return = wealth[-1] - 1.0

    years = len(returns) / TRADING_DAYS_PER_YEAR
    cagr = (
        wealth[-1] ** (1.0 / years) - 1.0
        if wealth[-1] > 0 and years > 0
        else np.nan
    )

    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    max_drawdown = drawdown.min()

    std = np.std(returns, ddof=1)
    sharpe = (
        np.mean(returns) / std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if std > 0
        else np.nan
    )

    return total_return, cagr, max_drawdown, sharpe


def circular_block_bootstrap(returns, rng, block_length):
    n = len(returns)
    sampled = []

    while len(sampled) < n:
        start = int(rng.integers(0, n))
        indices = (start + np.arange(block_length)) % n
        sampled.extend(returns[indices].tolist())

    return np.asarray(sampled[:n], dtype=float)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Run the original OOS evaluation first."
        )

    df = pd.read_csv(INPUT_FILE)

    if "daily_return" not in df.columns:
        raise ValueError(
            "Expected a daily_return column in the equity report."
        )

    returns = pd.to_numeric(
        df["daily_return"], errors="coerce"
    ).dropna().to_numpy()

    if len(returns) < BLOCK_LENGTH * 5:
        raise ValueError(
            "Too few daily observations for this block-bootstrap setup."
        )

    if np.any(returns <= -1):
        raise ValueError(
            "Daily return <= -100% found; inspect the input data first."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    observed = calculate_metrics(returns)
    samples = np.empty((N_BOOTSTRAPS, 4))

    for i in range(N_BOOTSTRAPS):
        boot = circular_block_bootstrap(
            returns, rng, BLOCK_LENGTH
        )
        samples[i] = calculate_metrics(boot)

    names = [
        "total_return",
        "annualized_cagr",
        "max_drawdown",
        "annualized_sharpe",
    ]

    rows = []

    for j, name in enumerate(names):
        values = samples[:, j]
        values = values[np.isfinite(values)]

        rows.append({
            "metric": name,
            "observed": observed[j],
            "bootstrap_p2_5": np.percentile(values, 2.5),
            "bootstrap_median": np.percentile(values, 50),
            "bootstrap_p97_5": np.percentile(values, 97.5),
            "bootstrap_probability_positive": (
                np.mean(values > 0)
            ),
        })

    report = pd.DataFrame(rows)
    output_file = OUTPUT_DIR / "oos_bidirectional_bootstrap.csv"
    report.to_csv(output_file, index=False)

    print("=" * 72)
    print("OOS BLOCK BOOTSTRAP")
    print("=" * 72)
    print(f"Observations: {len(returns)}")
    print(f"Block length: {BLOCK_LENGTH}")
    print(f"Bootstrap repetitions: {N_BOOTSTRAPS}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    print(
        report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()
    print(f"Saved: {output_file}")
    print(
        "\nInterpretation: these are exploratory resampling ranges, "
        "not guaranteed future-performance intervals."
    )


if __name__ == "__main__":
    main()