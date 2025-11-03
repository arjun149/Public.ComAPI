# Crypto Weekly Portfolio Allocation Prototype

This prototype implements baseline portfolio allocation methods for a weekly-rebalanced crypto portfolio, written in Python.

Structure
- data/                - cached raw and processed data (CSV)
- notebooks/           - notebooks for exploration and backtests
- src/
  - data_loader.py     - fetches OHLCV weekly data from Public.com API and caches CSVs
  - preprocess.py      - computes returns, rolling vol, covariances, features
  - optimizers.py      - equal-weight, min-variance, mean-variance (cvxpy)
  - backtester.py      - weekly rebalancer, cost/slippage model, metrics
  - utils.py           - config loader, plotting utilities, metrics
- config.yaml          - main configuration (universe, dates, costs, leverage, target vol)
- requirements.txt
- results/             - backtest outputs, plots, reports

Getting started
1) Create virtualenv and install dependencies:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Configure:
   Edit config.yaml to set the asset universe, date range, costs and risk target.

3) Fetch and cache data:
   python -m src.data_loader --config config.yaml

4) Run baseline backtests:
   python -m src.backtester --config config.yaml --methods equal,minvar,meanvar

Default assumptions
- Weekly rebalancing (close), start capital = 100,000 USD
- Universe = top cryptos by market cap (configurable)
- Max gross leverage = 2x, long-short allowed
- Transaction cost default = 0.1% round-trip
- Target portfolio volatility = 10% annualized

NEW (11/2/25): Updating RL model with a market simulator environment, training off-the-shelf continous RL policy, and evaluation, backtesting, and metrics. 
