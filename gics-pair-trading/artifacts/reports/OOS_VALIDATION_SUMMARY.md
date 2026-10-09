# GICS Pair Trading — Out-of-Sample Validation Report

**Status:** Research-stage validation completed; not approved for live capital deployment.

## 1. Executive Summary

The GICS pair-trading strategy produced a positive out-of-sample result over the evaluated period, but the available evidence does not yet establish a robust, repeatable trading edge.

The validation process covered execution integrity, directional strategy comparison, transaction-cost and borrow-cost stress testing, trade-concentration analysis, block bootstrap resampling, and temporal stability.

**Current decision: retain the strategy as a research candidate. Do not deploy real capital based on these results alone.**

## 2. Baseline OOS Performance

| Metric                   |                   Result |
| ------------------------ | -----------------------: |
| Evaluation period        | 2025-01-14 to 2026-10-02 |
| Recorded observations    |                      441 |
| Starting equity          |   Approximately $100,000 |
| Ending equity            |              $103,976.14 |
| Total return             |                  +3.976% |
| CAGR                     |                    2.25% |
| Annualized Sharpe ratio  |                    0.762 |
| Annualized Sortino ratio |                    1.105 |
| Maximum drawdown         |                  -3.229% |
| Executed trades          |                       57 |
| Duplicate-pair skips     |                        3 |

These results describe the historical simulation and should not be interpreted as expected live performance.

## 3. Execution Integrity

The recorded execution audit reported zero violations in the following checks:

* Entry occurred after the signal.
* Entry date followed the signal under the T+1 rule.
* Entry prices matched the specified opening-price convention.
* Exit dates matched the specified holding horizon.
* Exit prices matched the specified closing-price convention.
* No required execution prices were missing.

Passing these checks supports implementation consistency under the tested assumptions. It does not establish that real-world fills, short availability, borrow costs, or market impact will match the simulation.

## 4. Directional Comparison

| Strategy      |  Return | Max drawdown | Sharpe | Executed trades |
| ------------- | ------: | -----------: | -----: | --------------: |
| Bidirectional | +3.976% |      -3.229% |  0.762 |              57 |
| LONG_A only   | +0.144% |      -3.185% |  0.048 |              37 |
| LONG_B only   | +2.672% |      -2.072% |  0.834 |              23 |

The LONG_B-only result is promising but is based on a limited number of trades. It is not sufficient evidence to permanently remove LONG_A or to select LONG_B as the production strategy. Directional selection requires independent validation.

## 5. Cost and Borrow Stress Testing

The bidirectional strategy remained profitable under several tested scenarios:

| Transaction cost | Annual borrow assumption | Allocation assumption | Total return |
| ---------------- | -----------------------: | --------------------: | -----------: |
| 5 bps            |                       3% |                    5% |      +3.976% |
| 10 bps           |                       3% |                    5% |      +3.678% |
| 20 bps           |                       3% |                    5% |      +3.083% |
| 5 bps            |                      10% |                    5% |      +2.516% |
| 5 bps            |                      20% |                    5% |      +0.460% |
| 30 bps           |                      20% |                    5% |      -0.980% |
| 30 bps           |                      20% |                   10% |      -2.112% |

These are modelled scenarios, not observed live trading costs. The allocation assumption must be interpreted using the backtest's specific capital and exposure definitions. Portfolio gross exposure, overlapping positions, and margin requirements must be independently verified before any live deployment.

## 6. Concentration and Statistical Uncertainty

The concentration analysis showed that removing the top five winning trades changed the full-sample return to approximately -0.222%. This indicates meaningful dependence on a small number of profitable observations.

A circular block bootstrap using 20-observation blocks and 5,000 resamples produced the following exploratory results:

| Metric            | Observed | Bootstrap 2.5th percentile | Median | Bootstrap 97.5th percentile |
| ----------------- | -------: | -------------------------: | -----: | --------------------------: |
| Total return      |    3.98% |                     -3.22% |  4.03% |                      11.61% |
| Annualized CAGR   |    2.25% |                     -1.85% |  2.28% |                       6.48% |
| Annualized Sharpe |    0.762 |                     -0.615 |  0.775 |                       2.165 |

Approximately 84.9% of bootstrap samples had positive total returns. This is a resampling statistic, not the probability that the strategy will make money in the future and not proof of statistical significance.

The bootstrap resampled realized portfolio returns rather than regenerating prices, signals, pair selection, and execution. It therefore does not capture all sources of strategy uncertainty.

## 7. Temporal Stability

| Period                 |  Return | Annualized Sharpe | Maximum drawdown |
| ---------------------- | ------: | ----------------: | ---------------: |
| 2025                   | +2.587% |             0.993 |          -1.829% |
| 2026 through October 2 | +1.354% |             0.538 |          -2.629% |
| 2026 Q1                | -1.393% |            -1.958 |          -2.433% |
| 2026 Q2                | -0.104% |            -0.080 |          -2.629% |
| 2026 Q3                | +2.665% |             3.409 |          -0.992% |

The strategy experienced a material deterioration during the first half of 2026 followed by a strong recovery in Q3. The Q4 observation count was only two and its quarterly Sharpe is not interpretable.

Across the 382 valid rolling 60-observation windows, 26.70% had negative Sharpe and 26.96% had negative compounded returns. The latest window was positive, but overlapping windows are highly dependent and do not represent independent confirmations.

Trade-level attribution showed that Q3 profits were concentrated in LONG_A trades and a small number of winners. Quarterly trade contributions should not be equated with daily portfolio returns without reconciling holding-period P&L and exit-date attribution.

## 8. Known Limitations

The current validation does not fully establish:

* Consistent corporate-action, dividend, and adjusted-price treatment.
* Realistic short-sale availability, borrow recalls, and short dividend obligations.
* Actual commissions, bid-ask spreads, slippage, market impact, and opening/closing execution feasibility.
* Margin requirements, financing costs, and total gross exposure during overlapping trades.
* Stability of cointegration relationships and hedge ratios through structural changes.
* Robustness across independently specified market regimes and unseen future periods.
* Independent statistical significance after considering pair reuse, parameter selection, and multiple testing.

## 9. Decision and Next Stage

**Decision: research continuation, not live deployment.**

The next stage should prioritize operational realism and independent validation rather than tuning parameters to improve the historical result.

Recommended gates before considering real capital:

1. Reconcile daily portfolio P&L, realized trade contributions, and unrealized P&L.
2. Validate corporate actions, dividends, short costs, and execution assumptions.
3. Test pair and hedge-ratio stability and prevent look-ahead or survivorship bias.
4. Evaluate exposure limits, overlapping positions, and portfolio-level risk.
5. Run paper trading or shadow execution and compare simulated signals with executable prices.
6. Define acceptance criteria in advance and evaluate on new, untouched data.

The primary deliverable is a reproducible research and risk-validation pipeline. Positive historical returns are encouraging, but they are not sufficient evidence of a durable live-trading edge.
