# Pairs Trading Strategy in European Equity Markets

This project implements a fully functional **Pairs Trading Backtest** using daily price data of companies listed in the **STOXX Europe 600 Index**. It combines **statistical arbitrage**, **cointegration analysis**, and a **capital-tracked backtest** with realistic features such as transaction costs and short-selling fees.

The codebase is structured into three main components:
1. Statistical tests for identifying cointegrated pairs
2. A trading simulation engine
3. Performance evaluation and visualization

---



## Project Overview

**Pairs Trading** is a market-neutral strategy that involves identifying two historically correlated stocks, testing them for cointegration, and taking opposite positions when their price spread deviates significantly from the long-term mean.

This project:
- Identifies high-correlation stock pairs (≥ 0.9)
- Performs **ADF-based I(1) classification** to rule out stationarity
- Conducts the **Engle-Granger cointegration test** with optimal lag selection
- Generates trading signals based on the **Z-score** of the spread
- Simulates trades with:
  - Position sizing
  - Transaction costs (0.3%)
  - Short-selling fees (1% p.a.)
- Tracks and exports portfolio value over time

---

## File Overview
| **File Category**                      | **File**                              | **Description** | **Uses Data File(s)** |
|----------------------------------------|---------------------------------------|-----------------|-----------------------|
| **Time Series Data**                   | `EuroStoxx600_Weekdays.xlsx`          | Filtered Euro STOXX 600 price data (weekdays only), Bloomberg, Jun 2022–May 2025 | – |
|                                        | `EURO_STOXX_600_Index.xlsx`           | Historical Euro STOXX 600 index prices, STOXX Europe, Jun 2022–May 2025 | – |
|                                        | `EuroStoxx600_Weekdays_Sektor.xlsx`   | Euro STOXX 600 prices with sector classification, Bloomberg, Jun 2022–May 2025 | – |
|                                        | `Fama_French_Data.xlsx`               | Fama–French three-factor data, Kenneth R. French, Jun 2024–May 2025 | – |
| **Python Code**                        | `Pairs_Trading_Algo.py`               | Main pairs trading algorithm (cointegration & backtesting) | `EuroStoxx600_Weekdays.xlsx` |
|                                        | `Pairs_Trading_Algo_Sector.py`        | Sector-based pairs trading algorithm variant | `EuroStoxx600_Weekdays_Sektor.xlsx` |
|                                        | `EURO_STOXX_600.py`                   | Compares performance of `Pairs_Trading_Algo.py` with Euro STOXX 600 index | `EURO_STOXX_600_Index.xlsx` |
|                                        | `Fama_French_Analysis.py`              | Fama–French regression analysis of strategy returns | `Fama_French_Data.xlsx` |





---

## Key Features

| Module                      | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `load_data()`               | Loads and log-transforms Excel price data                                   |
| `classify_series()`         | Determines stationarity using ADF tests (I(0), I(1), or invalid)            |
| `run_engle_granger_test()`  | Identifies cointegrated pairs with BIC-optimized lag selection              |
| `generate_trading_signals()`| Creates entry/exit signals using Z-score thresholds                         |
| `run_trading_simulation()`  | Executes trades, tracks capital, calculates PnL, applies cost structure     |
| `analyze_costs()`           | Calculates total transaction and short-selling costs                        |
| `calculate_sharpe_ratio()`  | Evaluates risk-adjusted performance metrics                                 |
| Plotting functions          | Include drawdown, return histogram, capital allocation over time            |

---

## Output Files

The script produces several helpful outputs:
- `capital_tracking.csv`: Daily capital allocation (cash vs. invested)
- `portfolio_series.csv`: Total portfolio value over time
- `cointegration_results.csv`: Cointegrated pairs with test stats
- Visualizations: Drawdown, portfolio value, return distribution



## Parameters (set in main())

| Parameter                 | Value    | Description                                                                 |
|---------------------------|---------:|-------------------------------------------------------------------------------|
| `train_window`            | 518      | Number of trading days used for in-sample (training) estimation              |
| `initial_capital`         | 100000   | Starting capital (USD)                                                       |
| `z_score_threshold`       | 2        | Z-score threshold that triggers an entry signal                              |
| `min_cash_threshold`      | 10000    | Minimum cash required to open a new position (safety buffer)                 |
| `transaction_cost_rate`   | 0.003    | Transaction costs per side (0.3%) applied on entry and exit                  |
| `short_fee_rate`          | 0.01     | Annualized short-selling fee, prorated by holding period                     |
| `risk_free_rate_annual`   | 0.0431   | Annual risk-free rate used for Sharpe ratio calculation                      |
| `threshold`               | 0.9      | Minimum correlation the two log-price series must exceed                     |



