# Natural Gas Futures Strategy Backtest

## Overview

This project evaluates a rule-based natural gas futures mean-reversion strategy over:

**2021-09-17 → 2026-09-17**

The objective is not simply to maximize historical return, but to determine **where the strategy's economic edge actually comes from** and whether that edge survives across different natural-gas market regimes.

The backtest uses individual CME natural gas futures contracts rather than treating the Yahoo continuous contract as the tradable instrument.

### Core assumptions

* Initial capital: **$10,000**
* Market: CME Henry Hub Natural Gas futures
* Signal: official CME settlement data
* Execution: next settlement session's actual contract open
* Contract handling: individual futures contracts with explicit roll transitions
* Strategy: low-price mean reversion
* Backtest period: 2021-09-17 through 2026-09-17
* Closed trades: **38**
* Open trades at end: **0**

---

# 1. Main Result

The tested strategy generated:

| Metric                                  |                 Result |
| --------------------------------------- | ---------------------: |
| Closed trades                           |                     38 |
| Winning trades                          |                     38 |
| Losing trades                           |                      0 |
| Total P&L                               |          **+$243,670** |
| Initial capital                         |                $10,000 |
| Final capital                           | approximately $253,670 |
| Holding-period contribution             |           **-$26,310** |
| Roll-transition accounting contribution |          **+$149,220** |
| Final-exit contribution                 |          **+$120,760** |

The 100% historical win rate should **not** be interpreted as evidence that the strategy has a 100% future win rate.

The sample is relatively small and the profitability is strongly concentrated in particular market conditions.

---

# 2. What Actually Generated the Profit?

The economic attribution separates the strategy into:

1. Price-reversal return
2. Holding-period return
3. Roll-transition effects
4. Individual trade concentration
5. Market-regime dependence

The important distinction is that **roll-transition accounting is not automatically independent alpha**.

The detailed counterfactual analysis found:

* 100 roll records
* 78 feasible roll comparisons
* 18 unique feasible events
* Unique roll-timing effect: approximately **+$3,710**
* 11 positive timing effects
* 7 negative timing effects

Therefore roll timing contributed only about:

**1.52% of total strategy P&L**

This suggests that the large roll-transition accounting component should not be interpreted as a standalone source of trading edge.

---

# 3. Trade Concentration

The strategy does not appear to depend on one or two extraordinary trades.

Across the 38 closed trades:

* Average P&L: approximately **$6,412**
* Median P&L: approximately **$6,335**
* Largest single trade: approximately **$14,260**
* Largest trade contribution: approximately **5.85% of total P&L**

Therefore the historical result is distributed across many profitable trades rather than being explained by a single extreme outlier.

However, this does not eliminate regime risk.

---

# 4. Market-Regime Analysis

The most important finding is the distribution of profits across time.

Historical profitability was heavily concentrated in **2023–2024**, particularly 2024.

Approximately:

**2024 contributed $181,020, or about 74.3% of total P&L.**

There were no strategy trades in:

* 2021
* 2022
* 2025
* 2026

This is important.

The evidence therefore does **not** support the claim that low-price mean reversion is a continuously active edge across every natural-gas market environment.

Instead, the historical evidence is more consistent with a **regime-dependent strategy**.

The strategy becomes active when natural gas enters sufficiently depressed price conditions, but the frequency and profitability of those conditions vary substantially over time.

---

# 5. Economic Interpretation

The strategy's historical performance can be summarized as:

> **The main economic mechanism is buying unusually depressed natural-gas prices and benefiting when price subsequently reverts toward a higher level.**

The research does not show that:

> "Natural gas always reverts upward after becoming cheap."

A more defensible interpretation is:

> **Extreme low-price conditions can create profitable mean-reversion opportunities, but those opportunities are intermittent and highly dependent on the prevailing natural-gas market regime.**

This distinction is central to the strategy's risk assessment.

---

# 6. Robustness

The threshold robustness analysis tested multiple entry and exit combinations.

The results show a region of profitable parameter combinations rather than dependence on exactly one threshold.

This is stronger evidence than finding a single optimal parameter pair.

Nevertheless, parameter robustness within the same historical sample does not eliminate:

* regime risk
* sample-selection risk
* futures-roll effects
* execution assumptions
* transaction costs
* margin requirements
* slippage
* structural changes in the natural-gas market

---

# 7. What the Backtest Does NOT Prove

This research does not prove:

* future profitability
* a 100% future win rate
* that the strategy works in all regimes
* that historical roll accounting represents independent alpha
* that the strategy can be traded with zero transaction costs
* that $10,000 of initial capital is sufficient for live futures trading
* that historical natural-gas behavior will remain unchanged

The purpose of the project is **economic validation of the historical strategy**, not a claim of guaranteed future returns.

---

# 8. Final Strategy Assessment

The research supports the following conclusion:

### Supported

* Extreme low-price natural-gas conditions have historically produced profitable mean-reversion opportunities.
* The result is not explained by a single giant trade.
* The strategy has a reasonably broad profitable threshold region.
* Roll timing itself appears to contribute relatively little independent economic value.

### Not established

* A universal mean-reversion edge across all natural-gas regimes.
* Persistent profitability outside the historical opportunity periods.
* Future performance comparable to the 2023–2024 period.

### Key risk

**Regime dependence is the dominant unresolved risk.**

The most important question for future research is therefore not:

> "Which threshold produces the highest backtest return?"

but:

> **"Under what market conditions does the low-price mean-reversion edge exist?"**

---

# 9. Reproducibility

The project separates:

### Final research datasets

`data/`

The repository keeps only the final backtest, robustness, trade-attribution, roll-timing, and deferred-roll audit outputs.

### Final analysis scripts

`scripts/`

Only the final analysis scripts are included in the public repository. Raw market downloads, intermediate datasets, and exploratory scripts are excluded from GitHub.

The final audit can be reproduced from the stored trade and attribution datasets without rerunning every historical experiment.

---

# 10. Research Status

**Status: Historical strategy audit completed.**

The project should now be treated as a completed research milestone rather than an endlessly optimized backtest.

Future work, if performed, should be separated into a new research phase focused on:

* out-of-sample validation
* walk-forward testing
* transaction costs and slippage
* volatility-scaled position sizing
* regime identification
* live/paper-trading validation

Those tests should not be mixed into the historical audit presented here.
