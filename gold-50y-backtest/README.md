# Gold & Silver 1968–2026 — Final Economic Audit

## Research Objective

This project performs a long-horizon quantitative economic audit of **Gold and Silver** from April 1968 through September 2026.

The objective is not to find the single best-performing parameter.

Instead, the research asks:

> **What actually drives Gold and Silver returns, which signals contain repeatable information, and which apparent edges disappear after out-of-sample testing?**

The analysis is organized into four phases:

* **Phase 1 — Buy & Hold Baseline**
* **Phase 2 — US 10Y Macro Regime**
* **Phase 3 — Gold/Silver Ratio Relative Value**
* **Phase 4 — Final Strategy Economic Audit**

The emphasis is on:

* long-term compounding
* maximum drawdown
* risk-adjusted returns
* regime dependence
* out-of-sample robustness
* concentration of returns
* economic interpretation
* avoiding parameter overfitting

---

# 1. Dataset

## Master Dataset

Final locked dataset:

```text
data/gold_silver_master_1968_2026_macro_sp500.csv
```

Coverage:

```text
1968-04-01 → 2026-09-22
```

Rows:

```text
15,240
```

Sources:

* Gold historical series
* Silver historical series
* Perth Mint Gold
* Perth Mint Silver
* FRED AUD/USD
* FRED US 10-Year Treasury Yield
* S&P 500 Price Return
* S&P 500 Total Return

### Data Integrity

The final master dataset passed:

```text
20 checks
0 failures
```

Important characteristics:

* no artificial interpolation
* no forward filling of Gold/Silver prices
* source boundary explicitly preserved
* 2021-04-07 → 2021-09-22 is retained as a genuine source gap
* Gold/Silver ratio calculated from Gold and Silver prices
* Perth Mint AUD prices converted to USD using FRED AUD/USD
* weekend/holiday FX values forward-filled only for currency conversion
* US 10Y and S&P 500 data are not forced onto a synthetic common trading calendar

The master dataset is **LOCKED**.

---

# 2. Phase 1 — Buy & Hold Baseline

## Objective

Establish the economic baseline before introducing any timing model.

Strategies:

* Gold Buy & Hold
* Silver Buy & Hold
* Gold/Silver 50/50
* Gold/Silver 60/40
* Gold/Silver 40/60

Period:

```text
1968-04-01 → 2026-09-22
```

Common valid tradable observations:

```text
14,605
```

Approximate duration:

```text
58.48 years
```

## Results

| Strategy          |      CAGR | Max Drawdown | Sharpe | Sortino |     Final $10k |
| ----------------- | --------: | -----------: | -----: | ------: | -------------: |
| **Gold**          | **8.45%** |      -70.26% |   0.52 |    0.70 | **$1,149,025** |
| Silver            |     5.94% |      -92.83% |   0.34 |    0.46 |       $291,466 |
| Gold/Silver 50/50 |     8.18% |      -80.97% |   0.44 |    0.60 |       $992,977 |
| Gold/Silver 60/40 |     8.39% |      -78.16% |   0.47 |    0.64 |     $1,114,567 |
| Gold/Silver 40/60 |     7.89% |      -83.97% |   0.42 |    0.57 |       $847,284 |

## Phase 1 Findings

### Gold

Gold was the strongest long-term compounder in the tested universe.

Its historical economic role is therefore closer to:

> **Core long-duration precious-metal exposure**

rather than a tactical trading vehicle.

### Silver

Silver produced substantially lower long-term CAGR while experiencing a dramatically deeper historical drawdown.

Its role is better described as:

> **High-beta cyclical exposure to the precious-metals complex**

rather than a direct substitute for Gold.

### Diversification

A Gold/Silver mix reduced some concentration in Gold but did not eliminate the severe drawdowns inherent in precious metals.

The 60/40 Gold/Silver portfolio produced:

```text
8.39% CAGR
-78.16% MDD
```

which was close to Gold's long-term CAGR but still carried substantial drawdown risk.

---

# 3. Phase 2 — US 10Y Macro Regime

## Objective

Test whether changes in the US 10-Year Treasury yield can provide a robust regime signal for switching between Gold and equities.

The key question:

> **Does interest-rate direction contain enough information to generate persistent allocation alpha?**

Signals tested:

* US10Y 3-month change
* US10Y 12-month change
* combined 3M + 12M regime

The final strategy construction uses a **one-period lagged allocation signal**, eliminating same-day look-ahead.

---

# 4. Phase 2C — True Out-of-Sample Results

## Full Period

| Strategy          |      CAGR |         MDD |   Sharpe |  Sortino | Final $10k |
| ----------------- | --------: | ----------: | -------: | -------: | ---------: |
| Gold B&H          |     7.30% |     -68.93% |     0.46 |     0.61 |   $615,810 |
| S&P 500 Price B&H |     7.85% |     -52.40% |     0.56 |     0.72 |   $827,743 |
| Gold/S&P 50/50    | **7.85%** | **-40.38%** | **0.70** | **0.94** |   $829,860 |
| US10Y 12M Switch  |     9.66% |     -54.73% |     0.59 |     0.78 | $2,201,103 |
| US10Y 3M Switch   |     6.54% |     -62.58% |     0.44 |     0.57 |   $405,379 |
| US10Y 3M+12M B    |    10.24% |     -66.51% |     0.62 |     0.83 | $2,989,997 |

The 12M switch produced:

```text
+1.81 percentage points CAGR
```

versus the Gold/S&P 50/50 benchmark.

However, it also produced a substantially deeper drawdown:

```text
-54.73% vs -40.38%
```

and lower Sharpe:

```text
0.59 vs 0.70
```

---

# 5. Phase 2D — Attribution and Robustness

The apparent advantage of the US10Y 12M signal does not remain equally strong across historical periods.

Excess CAGR versus Gold/S&P 50/50:

| Period              |  Excess CAGR |
| ------------------- | -----------: |
| Full period         | **+1.81 pp** |
| Excluding 1970–1982 |     +0.44 pp |
| Excluding 1970s     |     +0.60 pp |
| Excluding 1980s     |     +1.17 pp |
| 1983–2026           |     +0.67 pp |
| **1988–2026**       | **-0.28 pp** |
| **2000–2026**       | **-1.17 pp** |

The annual attribution is also highly uneven.

Large positive years included:

```text
1973  +71.79 pp
1979  +45.69 pp
1987  +41.12 pp
1974  +33.79 pp
1978  +26.19 pp
```

while large negative years included:

```text
1975  -26.51 pp
1972  -24.23 pp
2008  -20.78 pp
2002  -15.72 pp
2016  -13.55 pp
2026   -9.65 pp
```

This indicates that a meaningful portion of the historical benefit came from specific macro regimes rather than a stable cross-era relationship.

## Total Return Check

For 1988–2026, replacing S&P Price Return with S&P Total Return produced:

| Strategy              |      CAGR |
| --------------------- | --------: |
| Gold                  |     4.31% |
| S&P 500 Total Return  |    10.63% |
| Gold/S&P TR 50/50     | **8.14%** |
| US10Y 12M Switch + TR |     6.85% |

The 12M switch therefore underperformed the 50/50 Total Return benchmark by:

```text
-1.28 pp CAGR
```

## Phase 2 Conclusion

US10Y direction contains useful **macro regime information**, but the evidence does not support treating it as a robust standalone alpha engine.

The signal is therefore classified as:

> **Macro Overlay — not Core Alpha**

The 3-month signal is rejected as a core timing mechanism.

The combined 3M+12M signal is also rejected as a core strategy because the higher CAGR comes with very deep drawdown and substantial regime dependence.

---

# 6. Phase 3 — Gold/Silver Ratio Relative Value

## Objective

Test whether the Gold/Silver Ratio (GSR) provides a repeatable relative-value signal for Silver versus Gold.

Definition:

```text
GSR = Gold / Silver
```

Full historical distribution:

| Statistic |    GSR |
| --------- | -----: |
| Minimum   |  14.01 |
| P10       |  31.60 |
| P20       |  38.21 |
| Median    |  62.55 |
| P80       |  78.32 |
| P90       |  85.17 |
| Maximum   | 123.49 |

Current value on 2026-09-22:

```text
65.79
```

---

# 7. Phase 3A — Full-Sample GSR Event Study

The initial event study examined future:

```text
Silver return - Gold return
```

following different GSR regimes.

For Extreme High GSR (P90+):

| Horizon | Mean Relative Return |  Median | Win Rate |
| ------- | -------------------: | ------: | -------: |
| 21d     |               +1.60% |  +0.88% |    59.2% |
| 63d     |               +4.24% |  +3.40% |    68.9% |
| 126d    |               +9.87% |  +5.97% |    75.5% |
| 252d    |          **+18.34%** | +10.43% |    79.5% |

At first glance this appears to be a strong Silver mean-reversion signal.

However, full-sample quantiles contain future information.

Therefore this result is **not sufficient evidence of a tradable edge**.

---

# 8. Phase 3A.5 — Rolling Out-of-Sample GSR Test

Historical GSR quantiles were recalculated using only information available before each observation.

Minimum history:

```text
1260 observations
```

This removes the major look-ahead problem.

## Extreme High GSR — Rolling OOS

| Horizon | Mean Relative Return | Median |  Win Rate |
| ------- | -------------------: | -----: | --------: |
| 21d     |               +0.73% | -0.06% |     49.3% |
| 63d     |               +1.63% | +0.30% |     51.6% |
| 126d    |               +3.47% | -0.61% |     47.7% |
| 252d    |           **+6.48%** | +0.86% | **51.6%** |

The original full-sample 252-day result:

```text
+18.34%
```

collapsed to:

```text
+6.48%
```

after proper rolling OOS construction.

This is a major robustness finding.

## Phase 3A.5 Conclusion

> **Extreme GSR alone is not a robust standalone Silver-buy signal.**

It is better interpreted as a:

> **Relative-value state variable**

rather than a mechanical trading trigger.

---

# 9. Phase 3B — Conditional GSR Analysis

The next question was:

> **When does Extreme High GSR actually contain useful information?**

Conditions tested:

1. GSR rising/falling
2. Gold above/below 200-day MA
3. Silver above/below 200-day MA
4. Silver high/normal volatility
5. US10Y rising/falling

The strongest discriminator was:

> **Gold > 200-day moving average**

## Extreme GSR + Gold > MA200

| Horizon | Mean Silver−Gold | Median |  Win Rate |
| ------- | ---------------: | -----: | --------: |
| 21d     |           +0.83% | +0.24% |     53.1% |
| 63d     |           +2.23% | +2.55% |     59.3% |
| 126d    |       **+8.77%** | +4.54% | **63.2%** |
| 252d    |      **+17.50%** | +6.79% | **70.3%** |

Compared with Extreme GSR alone:

```text
126d: +3.47% → +8.77%
252d: +6.48% → +17.50%
```

This is the strongest conditional relationship identified in Phase 3.

---

# 10. Phase 3C — Conditional Regime Audit

Three predefined regimes were tested.

### R1

```text
Extreme GSR
```

### R2

```text
Extreme GSR
+
Gold > MA200
```

### R3

```text
Extreme GSR
+
Gold > MA200
+
Silver High Volatility
```

## Results

| Regime                   |     N |    21d |        63d |       126d |        252d |
| ------------------------ | ----: | -----: | ---------: | ---------: | ----------: |
| **R1 Extreme GSR**       | 4,728 | +0.73% |     +1.63% |     +3.47% |      +6.48% |
| **R2 + Gold > MA200**    | 1,120 | +0.83% |     +2.23% | **+8.77%** | **+17.50%** |
| **R3 + Silver High Vol** |   222 | +0.87% | **+4.25%** |     +3.30% |      +6.80% |

R3 does not improve the long-term relationship.

Its 252-day result falls from:

```text
R2: +17.50%
```

to:

```text
R3: +6.80%
```

The R3 252-day result is also highly concentrated:

```text
Top 10 observations ≈ 61% of total relative-return sum
```

Therefore R3 is rejected as a core regime.

---

# 11. Regime Stability

The most important warning from Phase 3 is regime dependence.

## R1 — Extreme GSR

252-day relative return:

| Era       |    Mean |
| --------- | ------: |
| 1973–1979 | +32.91% |
| 1980–1999 |  +0.12% |
| 2000–2011 | +34.44% |
| 2012–2019 |  -0.83% |
| 2020–2026 | +20.66% |

This clearly demonstrates that GSR mean reversion is not a universal constant.

## R2 — Extreme GSR + Gold > MA200

252-day relative return:

| Era       |             Mean |
| --------- | ---------------: |
| 1973–1979 |          +37.97% |
| 1980–1999 |           -6.62% |
| 2000–2011 | sample too small |
| 2012–2019 |          +12.10% |
| 2020–2026 |          +22.03% |

R2 is the strongest candidate identified, but it remains **regime-dependent**.

---

# 12. Phase 4 — Final Strategy Economic Audit

Phase 4 does not introduce new parameters.

It consolidates the evidence from Phases 1–3 into one economic framework.

## Final Comparison

| Component                      | Evidence                         | Economic Role                        | Robustness | Final Classification         |
| ------------------------------ | -------------------------------- | ------------------------------------ | ---------- | ---------------------------- |
| **Gold B&H**                   | 8.45% CAGR                       | Long-term precious-metal compounding | High       | **Core**                     |
| **Silver B&H**                 | 5.94% CAGR, -92.83% MDD          | High-beta cyclical exposure          | Medium     | **Satellite**                |
| **Gold/Silver 50/50**          | 8.18% CAGR                       | Precious-metal diversification       | High       | **Benchmark**                |
| **Gold/Silver 60/40**          | 8.39% CAGR                       | Gold-dominant diversified exposure   | High       | **Reference allocation**     |
| **US10Y 12M**                  | +1.81pp CAGR vs 50/50            | Macro regime information             | Medium-Low | **Overlay**                  |
| **US10Y 3M**                   | 6.54% CAGR                       | Short-term rate noise                | Low        | **Reject**                   |
| **US10Y 3M+12M**               | 10.24% CAGR, -66.51% MDD         | Gold-heavy regime exposure           | Low-Medium | **Reject as core**           |
| **Extreme GSR**                | OOS 252d +6.48%                  | Relative-value state                 | Medium-Low | **State variable**           |
| **GSR Direction**              | Rising/Falling little difference | Weak incremental information         | Low        | **Reject**                   |
| **Extreme GSR + Gold > MA200** | OOS 252d +17.50%                 | Conditional Silver/Gold RV regime    | Medium     | **Primary candidate regime** |
| **+ Silver High Vol**          | 252d +6.80%                      | Additional filtering                 | Low        | **Reject as core filter**    |

---

# 13. Final Economic Engine

After four phases, the research converges on a relatively simple economic structure.

```text
                    PRECIOUS METALS
                          │
             ┌────────────┴────────────┐
             │                         │
           GOLD                     SILVER
             │                         │
        Long-term Core          High-beta Satellite
             │                         │
             └────────────┬────────────┘
                          │
                    GSR Relative Value
                          │
                  Is GSR Extreme High?
                          │
                    ┌─────┴─────┐
                    │           │
                   No          Yes
                    │           │
                 Observe    Gold > MA200?
                                │
                         ┌──────┴──────┐
                         │             │
                        No            Yes
                         │             │
                    Weak signal    Candidate
                                   RV regime
```

The economic interpretation is:

### Gold

The primary long-term compounding asset.

### Silver

A higher-beta expression of the precious-metals cycle.

### GSR

A valuation/relative-value state variable.

### Gold MA200

The most informative trend confirmation discovered in the research.

### US10Y

Useful macro context, but not sufficiently stable to serve as the primary trading engine.

---

# 14. Current Market State — 2026-09-22

Current Gold/Silver Ratio:

```text
65.79
```

Historical thresholds:

```text
Median = 62.55
P80    = 78.32
P90    = 85.17
```

Therefore:

```text
65.79 > Median
65.79 < P80
65.79 < P90
```

The current GSR is elevated relative to the historical median, but it is **not an Extreme High GSR regime** under the project's predefined framework.

Therefore the Phase 3 R2 condition is not currently triggered by GSR alone.

---

# 15. Final Findings

## Finding 1 — Gold is the long-term core

Over nearly six decades:

```text
Gold CAGR = 8.45%
Silver CAGR = 5.94%
```

Gold also experienced a materially lower maximum drawdown than Silver.

The long-term evidence therefore supports treating Gold as the core precious-metal exposure.

---

## Finding 2 — Silver is high-beta, not a superior Gold

Silver's historical behavior is highly asymmetric.

It can dramatically outperform during precious-metals bull markets, but the downside can be extreme.

Historical Silver:

```text
MDD = -92.83%
```

Therefore Silver should be interpreted as a cyclical satellite rather than an equivalent core holding.

---

## Finding 3 — Diversification matters

Gold/S&P 500 50/50 produced:

```text
CAGR = 7.85%
MDD  = -40.38%
Sharpe = 0.70
Sortino = 0.94
```

This demonstrates that Gold's economic value is not only its standalone return.

Its diversification contribution can materially alter portfolio drawdown characteristics.

---

## Finding 4 — US10Y is information, not proven alpha

The US10Y 12M switch produced a strong full-sample CAGR:

```text
9.66%
```

but the benefit weakened or disappeared in later historical samples.

With S&P Total Return data from 1988 onward:

```text
12M Switch = 6.85%
50/50       = 8.14%
```

Therefore the US10Y signal should not be promoted into a universal timing strategy.

---

## Finding 5 — Extreme GSR alone is insufficient

Full-sample analysis initially suggested:

```text
+18.34% 252-day Silver−Gold
```

But proper rolling OOS testing reduced this to:

```text
+6.48%
```

with only:

```text
51.6% win rate
```

This is a classic example of why full-sample historical optimization can overstate apparent alpha.

---

## Finding 6 — Gold > MA200 is the most informative condition

Combining:

```text
Extreme GSR
+
Gold > MA200
```

produced:

```text
126d: +8.77%
252d: +17.50%
252d win rate: 70.3%
```

This is the strongest conditional relationship identified in the project.

However, it remains regime-dependent and should not be interpreted as a guaranteed trading signal.

---

## Finding 7 — More filters do not necessarily improve the model

Adding Silver High Volatility produced:

```text
R2 252d = +17.50%
R3 252d = +6.80%
```

The additional filter therefore destroyed much of the long-horizon relationship.

This provides evidence against continued parameter stacking and overfitting.

---

# 16. Final Strategy Classification

The final research framework is therefore:

```text
CORE
└── Gold Buy & Hold

SATELLITE
└── Silver / cyclical precious-metals exposure

RELATIVE VALUE
└── Gold/Silver Ratio

TREND CONFIRMATION
└── Gold > 200-day MA

MACRO CONTEXT
└── US 10Y 12-month direction

REJECTED AS CORE SIGNALS
├── US10Y 3-month switch
├── GSR direction
└── Excessive multi-condition filtering
```

---

# 17. What This Research Does NOT Claim

This project does **not** claim:

* that Gold will outperform in the future
* that Silver must mean-revert whenever GSR is high
* that US10Y direction creates permanent alpha
* that the R2 regime guarantees Silver outperformance
* that the historical CAGR represents a forecast
* that event-study returns equal executable portfolio returns

Phase 3 is an **event-study / conditional relative-value analysis**, not a fully executable futures strategy.

It does not yet model:

* transaction costs
* bid/ask spreads
* leverage
* futures margin
* contract rolls
* slippage
* position sizing
* stop-loss rules
* taxes
* financing costs

These limitations are particularly important for Silver because of its much higher volatility.

---

# 18. Reproducibility

Suggested project structure:

```text
gold-50y-backtest/
│
├── data/
│   ├── gold_silver_master_1968_2026_macro_sp500.csv
│   ├── gold_silver_long_history.csv
│   ├── gold_silver_perth_usd_2021_2026.csv
│   ├── phase2c_true_oos_annual_returns.csv
│   ├── phase2c_true_oos_daily_returns.csv
│   └── updated/
│
├── scripts/
│   ├── build_master_dataset.py
│   ├── phase1_buy_hold.py
│   ├── merge_dgs10_master.py
│   ├── phase2c_us10y_true_oos.py
│   ├── phase3a_gsr_event_study.py
│   ├── phase3a5_gsr_rolling_oos.py
│   ├── phase3b_gsr_conditional_edge.py
│   └── phase3c_gsr_conditional_regime_audit.py
│
└── README.md
```

The master dataset should remain locked after validation.

---

# 19. Bottom Line

The four-phase audit does **not** discover a magical Gold/Silver timing strategy.

Instead, it identifies a much more defensible economic structure:

> **Gold is the long-term core. Silver is the high-beta cyclical satellite. GSR is a relative-value state variable rather than a standalone trading signal. When GSR becomes extreme while Gold remains above its 200-day moving average, the historical evidence shows materially stronger subsequent Silver-vs-Gold relative performance. US10Y provides useful macro context, but its timing advantage is not stable enough to constitute standalone alpha.**

The most important lesson is methodological:

> **The strongest-looking full-sample signals became substantially weaker after proper out-of-sample testing. The goal of this project is therefore not to maximize historical CAGR, but to identify economic relationships that survive robustness checks.**

That is the central conclusion of the **Gold & Silver 1968–2026 Economic Audit**.
