# GICS Pair Trading Research

A Python research project for systematic US equity pair trading using GICS-related industry groupings, historical price data, cointegration and spread analysis, and out-of-sample (OOS) backtesting.

## Project Goals

* Maintain a configurable US equity universe.
* Download and validate historical OHLCV price data.
* Build aligned price and returns matrices.
* Generate and evaluate candidate equity pairs.
* Backtest pair-trading strategies with explicit execution assumptions.
* Evaluate directional exposure, transaction costs, borrowing costs, concentration risk, and temporal stability.

## Repository Structure

* `config/`: data-provider settings, ticker mappings, and universe configuration.
* `src/`: reusable data, price-quality, storage, update, and universe modules.
* `scripts/`: data audits, pair selection, backtesting, and OOS evaluation scripts.
* `tests/`: automated tests.
* `artifacts/reports/`: selected backtest outputs, validation tables, figures, and research summaries.
* `data/`: local market data; raw and processed data directories are excluded from Git.

## Research Status

The current research includes an OOS evaluation, execution-integrity checks, transaction-cost and borrowing-cost stress tests, concentration analysis, bootstrap resampling, and temporal stability analysis.

The current baseline OOS result is documented in `artifacts/reports/OOS_VALIDATION_SUMMARY.md`. Historical backtest results are research evidence, not proof of future profitability. The strategy has not been established as suitable for live trading.

## Setup

Use Python 3.12 if possible.

Create and activate a virtual environment in Windows CMD:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```cmd
py -m pip install -r requirements.txt
```

Run tests:

```cmd
py -m pytest
```

Run scripts from the repository root unless a script's instructions specify otherwise.

## Data and Reproducibility

Market data may change after downloads, and provider coverage can vary. Record the data window, configuration, execution assumptions, and output files when reproducing an experiment. Do not treat an OOS backtest as a guarantee of future returns.

## Disclaimer

This repository is for research and educational purposes only. It is not investment advice or a recommendation to buy or sell any security.
