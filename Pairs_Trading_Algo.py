"""
===============================================================================
 Author         : Philipp König
 Final Version  : 2025-08-14
===============================================================================

Description:
------------
Full implementation of a cointegration-based pairs trading strategy for 
European equities. The script performs:

1. **Data Loading & Preprocessing**:
   - Reads Euro STOXX 600 daily price data from Excel
   - Converts to log prices for statistical testing
   - Filters time series for minimum length

2. **Statistical Tests**:
   - Correlation filtering (threshold-based)
   - I(1) classification via Augmented Dickey-Fuller test
   - Engle-Granger cointegration testing with optimal lag selection (BIC)

3. **Trading Simulation**:
   - Spread computation and Z-score based entry/exit signals
   - Position sizing with capital tracking
   - Transaction costs and short-selling fees applied
   - Rolling portfolio value computation

4. **Performance Evaluation**:
   - Drawdown metrics (max & average)
   - Volatility, Sharpe, Calmar, Sterling ratios
   - Cost analysis
   - Visualization: returns histogram, drawdown plot, capital allocation over time

Dependencies:
-------------
- Python 3.12
- pandas
- numpy
- matplotlib
- statsmodels

Input Files:
------------
- EuroStoxx600_Weekdays.xlsx (daily close prices with 'Date' column)

Output Files:
-------------
- cointegration_results.csv   : Cointegrated pairs with regression parameters
- capital_tracking.csv        : Portfolio cash/invested history
- portfolio_series.csv        : Portfolio value time series

Notes:
------
Set these paramenters in def main():
    train_window = 
    initial_capital = 
    z_score_threshold = 
    min_cash_threshold = 
    transaction_cost_rate =
    short_fee_rate = 
    risk_free_rate_annual = 
    threshold = 
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as patches
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.stattools import adfuller
import os
import exchange_calendars as ecals
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import re


# =============================================================================
# SECTION 1 – STATISTICAL TESTS
# This section includes all preprocessing steps and statistical tests,
# including I(1) classification, correlation filtering, and cointegration testing.
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as patches
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.stattools import adfuller
import os
import exchange_calendars as ecals
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde



# =============================================================================
# SECTION 1 – STATISTICAL TESTS
# This section includes all preprocessing steps and statistical tests,
# including I(1) classification, correlation filtering, and cointegration testing.
# =============================================================================



train_window = 518


### Log Retruns ###
def get_local_filepath(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def load_data(filename):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    df = pd.read_excel(filepath)  
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna(axis=1)
    

    price_df = df.iloc[:, 1:].copy()  
    log_df = np.log(price_df) 
    log_df = log_df.dropna(axis=1, thresh=300)

    return price_df, log_df
    


### Correlation Matrix >= 0.9 ###
def find_high_corr_pairs(log_df, threshold):
    corr_matrix = log_df.corr(numeric_only=True)
    corr_pairs = corr_matrix.unstack()
    corr_df = pd.DataFrame(corr_pairs, columns=["Correlation"]).reset_index()
    corr_df.columns = ["Stock1", "Stock2", "Correlation"]
    corr_df = corr_df[corr_df["Stock1"] != corr_df["Stock2"]]
    corr_df["Pair"] = corr_df.apply(lambda row: tuple(sorted([row["Stock1"], row["Stock2"]])), axis=1)
    corr_df = corr_df.drop_duplicates(subset="Pair")
    high_corr_pairs = corr_df[corr_df["Correlation"] >= threshold].sort_values(by="Correlation", ascending=False)

    return high_corr_pairs.drop(columns=["Pair"])


# === 3. ADF-test; order of time series ===
def classify_series(log_df, min_len=50):
    I0_stocks, I1_stocks, invalid_stocks = [], [], []
    for i, stock in enumerate(log_df.columns):
        series = log_df[stock].dropna()
        if len(series) < min_len:
            invalid_stocks.append(stock)
            continue
        try:
            adf_level = adfuller(series, maxlag=10, autolag='BIC')
            if adf_level[1] < 0.05:
                I0_stocks.append(stock)
            else:
                adf_diff = adfuller(series.diff().dropna(), maxlag=10, autolag='BIC')
                if adf_diff[1] < 0.05:
                    I1_stocks.append(stock)
                else:
                    invalid_stocks.append(stock)
        except Exception as e:
            invalid_stocks.append(stock)
    return I0_stocks, I1_stocks, invalid_stocks




# === 4. best lag length ===
def get_best_lag(residuals, max_lag=None):
    T = len(residuals)
    if max_lag is None:
        max_lag = int(np.floor(12 * (T / 100) ** (1/4)))  

    best_bic = np.inf
    best_lag = 1

    delta_z = np.diff(residuals)  

    for p in range(1, max_lag + 1):

        delta_z_lagged = lagmat(delta_z, maxlag=p - 1, trim='both')
        z_lagged = residuals[p:]  


        y = delta_z[p - 1:]


        if len(z_lagged) != len(delta_z_lagged) or len(y) != len(z_lagged):
            continue
        if len(y) == 0 or np.var(y) < 1e-8:
            continue

        # === regression ===
        X = np.column_stack([z_lagged, delta_z_lagged])
        X = add_constant(X)

        try:
            model = OLS(y, X).fit()
            bic = model.bic
            if np.isfinite(bic) and bic < best_bic:
                best_bic = bic
                best_lag = p
        except Exception:
            continue

    return best_lag, best_bic



# === 5. ADF test for residuals (Engle-Granger, step 2) ===
def engle_granger_adf(residuals):
    p_opt, bic = get_best_lag(residuals)
    if p_opt < 1:
        return None, None, None
    delta_z = np.diff(residuals)
    if p_opt == 1:
        z_lagged = residuals[1:-1]
        y = delta_z[1:]
        if len(z_lagged) != len(y):
            return None, None, None
        X = np.column_stack([z_lagged])
    else:
        z_lagged = residuals[p_opt:-1]
        delta_z_lagged = lagmat(delta_z, maxlag=p_opt - 1, trim='both')
        y = delta_z[p_opt - 1:]
        if len(z_lagged) != len(delta_z_lagged) or len(y) != len(z_lagged):
            return None, None, None
        X = np.column_stack([z_lagged, delta_z_lagged])
    X = add_constant(X)
    model = OLS(y, X).fit()
    gamma_hat = model.params[1]
    SE_gamma = model.bse[1]
    test_stat = gamma_hat / SE_gamma
    return test_stat, p_opt, model


# === 6. Engle-Granger test for all pairs ===
def run_engle_granger_test(log_df, high_corr_pairs, I1_stocks, crit_value=-3.44):
    cointegrated_pairs = []
    for _, row in high_corr_pairs.iterrows():
        stock1 = row["Stock1"]
        stock2 = row["Stock2"]
        if stock1 not in I1_stocks or stock2 not in I1_stocks:
            continue
        y_full = log_df[stock1].dropna()
        x_full = log_df[stock2].dropna()
        common_idx = y_full.index.intersection(x_full.index)
        y_full = y_full.loc[common_idx]
        x_full = x_full.loc[common_idx]
        if len(y_full) < train_window + 10:
            continue
        y_train = y_full.iloc[:train_window]
        x_train = x_full.iloc[:train_window]
        try:
            x_const = add_constant(x_train)
            reg_model = OLS(y_train, x_const).fit()
            residuals = reg_model.resid
            test_stat, opt_lag, adf_model = engle_granger_adf(residuals)
            if test_stat is not None and test_stat < crit_value:
                cointegrated_pairs.append({
                    "Stock1": stock1,
                    "Stock2": stock2,
                    "ADF_stat": test_stat,
                    "Lag": opt_lag,
                    "Beta": reg_model.params[1],
                    "Alpha": reg_model.params[0]
                })
        except Exception as e:
            print(f"Fehler bei Paar {stock1} - {stock2}: {e}")
            continue
    return pd.DataFrame(cointegrated_pairs)







# =============================================================================
# SECTION 2 – TRADING SIMULATION
# This section contains the core pairs trading logic, including entry/exit rules,
# position sizing, capital tracking, and transaction cost handling.
# =============================================================================


def compute_spread(y, x, alpha, beta):
    return y - alpha - beta * x

def generate_trading_signals(spread, mu=None, sigma=None, z_score_threshold=2):
    if mu is None:
        mu = np.mean(spread)
    if sigma is None or sigma < 1e-4:
        sigma = np.nan

    z_score = (spread - mu) / sigma
    signals = []
    position = 0
    z_reset = True

    for z in z_score:
        if position == 0:
            if z_reset:
                if z > z_score_threshold:
                    signals.append(-1)
                    position = -1
                    z_reset = False
                elif z < -z_score_threshold:
                    signals.append(1)
                    position = 1
                    z_reset = False
                else:
                    signals.append(0)
            else:
                signals.append(0)
        elif position == 1:
            if z >= -0.5:
                signals.append(0)
                position = 0
                z_reset = False
            else:
                signals.append(1)
        elif position == -1:
            if z <= 0.5:
                signals.append(0)
                position = 0
                z_reset = False
            else:
                signals.append(-1)

        if position == 0 and -0.5 < z < 0.5:
            z_reset = True

    return np.array(signals), pd.Series(z_score, index=spread.index)



def run_trading_simulation(
    log_df, price_df, 
    cointegration_results,
    train_window, 
    initial_capital,
    z_score_threshold, 
    min_cash_threshold,
    transaction_cost_rate,
    short_fee_rate
):
    cash = initial_capital
    executed_trades = []
    open_positions = []
    portfolio_values = defaultdict(lambda: cash)

    # === trading signals ===
    pair_signals = []
    for _, row in cointegration_results.iterrows():
        s1, s2 = row["Stock1"], row["Stock2"]
        alpha, beta = row["Alpha"], row["Beta"]

        if s1 not in log_df.columns or s2 not in log_df.columns:
            continue

        y = log_df[s1].dropna()
        x = log_df[s2].dropna()
        common_index = y.index.intersection(x.index)

        y_full = y.loc[common_index]
        x_full = x.loc[common_index]

        if len(y_full) < train_window + 10:
            continue  

        y_train = y_full.iloc[:train_window]
        x_train = x_full.iloc[:train_window]
        y_t = y_full.iloc[train_window:]
        x_t = x_full.iloc[train_window:]
        idx = y_t.index

        # Spread and Z-score based on training data
        spread_train = compute_spread(y_train, x_train, alpha, beta)
        mu = spread_train.mean()
        sigma = spread_train.std()

        spread_test = compute_spread(y_t, x_t, alpha, beta)
        signals, z_scores = generate_trading_signals(spread_test, mu=mu, sigma=sigma, z_score_threshold=z_score_threshold)

        pair_signals.append({
            "Pair": f"{s1}-{s2}",
            "Alpha": alpha,
            "Beta": beta,
            "Y_T": y_t,
            "X_T": x_t,
            "Spread": spread_test,
            "Z_Score": z_scores,
            "Signals": pd.Series(signals, index=idx)
        })

    dates = log_df.index[train_window:]

    def evaluate_open_positions_value(open_positions, current_date):
        value = 0
        for pos in open_positions:
            if current_date not in pos["Y_T"].index or current_date not in pos["X_T"].index:
                continue
            try:
                price_a = np.exp(pos["Y_T"].loc[current_date])
                price_b = np.exp(pos["X_T"].loc[current_date])
            except:
                continue

            entry_a = pos["Entry Price A"]
            entry_b = pos["Entry Price B"]
            direction = pos["Direction"]
            capital_long = pos["Invested Long"]
            capital_short = pos["Invested Short"]

            if direction == 1:
                pnl = capital_long * np.log(price_a / entry_a) + capital_short * np.log(entry_b / price_b)
            else:
                pnl = capital_short * np.log(price_b / entry_b) + capital_long * np.log(entry_a / price_a)

            value += pos["Invested"] + pnl
        return value

    for current_date in dates:
        # === closing positions ===
        for pos in open_positions[:]:
            if pos["Exit Date"] <= current_date:
                if current_date not in pos["Y_T"].index or current_date not in pos["X_T"].index:
                    continue

                exit_a = np.exp(pos["Y_T"].loc[current_date])
                exit_b = np.exp(pos["X_T"].loc[current_date])

                direction = pos["Direction"]
                entry_a = pos["Entry Price A"]
                entry_b = pos["Entry Price B"]
                capital_long = pos["Invested Long"]
                capital_short = pos["Invested Short"]

  
                shares_long = capital_long / entry_a
                shares_short = capital_short / entry_b

                # === PnL-Calculation 
                if direction == 1:
                    pnl = capital_long * np.log(exit_a / entry_a) + capital_short * np.log(entry_b / exit_b)
                else:
                    pnl = capital_short * np.log(exit_b / entry_b) + capital_long * np.log(entry_a / exit_a)

                # === Exit transaktions costen 
                exit_volume_long = shares_long * exit_a
                exit_volume_short = shares_short * exit_b
                exit_costs = (exit_volume_long + exit_volume_short) * transaction_cost_rate

                # === Shortselling fees
                entry_date = pos["Entry Date"]
                days_held = (current_date - entry_date).days
                short_fee = capital_short * short_fee_rate * (days_held / 252)

                cash += pos["Invested"] + pnl - exit_costs - short_fee

                executed_trades.append({
                    **pos,
                    "PnL": pnl,
                    "Exit Date": current_date,
                    "Days Held": days_held,
                    "Short Fee": short_fee,
                    "Exit Price A": exit_a,
                    "Exit Price B": exit_b,
                    "Exit Volume Long": exit_volume_long,
                    "Exit Volume Short": exit_volume_short,
                    "Exit Costs": exit_costs,
                    "Long Stock": pos["Y_T"].name if direction == 1 else pos["X_T"].name,
                    "Short Stock": pos["X_T"].name if direction == 1 else pos["Y_T"].name,
                })

                open_positions.remove(pos)

        # === open new positions ===
        for pair in pair_signals:
            sig_series = pair["Signals"]
            if current_date not in sig_series:
                continue
            signal = sig_series.loc[current_date]
            if signal == 0:
                continue
            if any(p["Pair"] == pair["Pair"] for p in open_positions):
                continue
            if cash < min_cash_threshold:
                continue

            invest_amount = 10000
            if cash < invest_amount:
                continue

            capital_a = invest_amount * 0.5
            capital_b = invest_amount * 0.5

            if signal == 1:
                invested_long = capital_a       # Long A, Short B
                invested_short = capital_b
            else:
                invested_long = capital_b       # Long B, Short A
                invested_short = capital_a

            y_t, x_t = pair["Y_T"], pair["X_T"]
            price_a = np.exp(y_t.get(current_date, np.nan))
            price_b = np.exp(x_t.get(current_date, np.nan))
            if np.isnan(price_a) or np.isnan(price_b):
                continue

            exit_idx = sig_series[(sig_series.index > current_date) & (sig_series == 0)].index
            if len(exit_idx) == 0:
                continue
            exit_date = exit_idx[0]

            # === Entry transaktions costs ===
            entry_costs = (invested_long + invested_short) * transaction_cost_rate

            open_positions.append({
                "Entry Date": current_date,
                "Exit Date": exit_date,
                "Direction": signal,
                "Invested": invested_long,
                "Pair": pair["Pair"],
                "Y_T": y_t,
                "X_T": x_t,
                "Entry Price A": price_a,
                "Entry Price B": price_b,
                "Invested Long": invested_long,
                "Invested Short": invested_short,
                "Beta": beta,
                "Entry Costs": entry_costs
            })

            cash -= (invested_long + entry_costs)

        try:
            market_value = evaluate_open_positions_value(open_positions, current_date)
        except Exception:
            market_value = 0

        total_value = cash + market_value
        portfolio_values[current_date] = total_value

        if 'capital_tracking' not in locals():
            capital_tracking = []

        capital_tracking.append({
            "Date": current_date,
            "Cash": cash,
            "Invested": market_value,
            "Total": total_value,
            "Open Positions": len(open_positions)
        })

    df_trades = pd.DataFrame(executed_trades)
    df_trades["Return (%)"] = (df_trades["PnL"] / df_trades["Invested"]) * 100
    portfolio_series = pd.Series(portfolio_values)
    portfolio_series.sort_index(inplace=True)


    capital_tracking_df = pd.DataFrame(capital_tracking)
    capital_tracking_df.to_csv(get_local_filepath("capital_tracking.csv"), index=False)

    return df_trades, portfolio_series, pair_signals






# =============================================================================
# SECTION 3 – PERFORMANCE EVALUATION
# This section covers portfolio statistics, risk metrics (volatility, Sharpe),
# cost analysis, drawdown, and visualizations of portfolio and trading behavior.
# =============================================================================



def analyze_costs(df_trades, transaction_cost_rate, initial_capital):

    # === Calculation ===
    df_trades["Transaction_Costs_Total"] = 2 * (df_trades["Invested Long"] + df_trades["Invested Short"]) * transaction_cost_rate
    total_transaction_costs = df_trades["Transaction_Costs_Total"].sum()
    total_short_fees = df_trades["Short Fee"].sum()
    total_pnl = df_trades["PnL"].sum()

    
    
    
    return total_transaction_costs, total_short_fees, total_pnl




def plot_daily_log_returns(portfolio_series, bins=25):
    """
    Plot a histogram and density of the daily log returns of the portfolio.
    """
    log_returns = np.log(portfolio_series / portfolio_series.shift(1)).dropna()

    plt.figure(figsize=(10, 6))

    # Histogram
    plt.hist(log_returns, bins=bins, color="#00236F", edgecolor='black', alpha=0.6, density=True)

    # Density estimate
    density = gaussian_kde(log_returns)
    xs = np.linspace(log_returns.min(), log_returns.max(), 200)
    plt.plot(xs, density(xs), color="darkred", linewidth=2, label="Density Estimate")
    plt.xlabel("Daily Log Return", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()


    return log_returns


def plot_drawdown(portfolio_series):
    """
    Plot the drawdown of the portfolio over time.
    """

    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series / running_max) - 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(drawdown, color='red')
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.set_xlim(drawdown.index.min(), drawdown.index.max())
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    return drawdown


def calculate_drawdown(portfolio_series):
    """
    Calculates the average drawdown based on the portfolio history.
    """
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series / running_max) - 1
    in_drawdown = drawdown < 0
    dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    drawdown_phases = drawdown[in_drawdown].groupby(dd_groups[in_drawdown])
    drawdown_mins = drawdown_phases.min()
    avg_drawdown = drawdown_mins.mean()

    max_drawdown = drawdown.min()

    return avg_drawdown, max_drawdown

    

def plot_portfolio_value(portfolio_series):
    """
    Plot the portfolio value over time.
    """
    plt.figure(figsize=(10, 6))

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()



  
def calculate_sharpe_ratio(risk_free_rate_annual, portfolio_series, max_drawdown, avg_drawdown):
    n_days = len(portfolio_series)
    
    log_total_return = (portfolio_series.iloc[-1] / 100000) -1
    annual_return = log_total_return


    log_returns = np.log(portfolio_series / portfolio_series.shift(1)).dropna()
    annual_volatility = log_returns.std() * np.sqrt(252)

    # Sharpe Ratio
    excess_return = annual_return - risk_free_rate_annual
    sharpe_ratio = excess_return / annual_volatility

    # === Calmar Ratio ===
    calmar_ratio = excess_return / abs(max_drawdown)

    # === Sterling Ratio ===
    sterling_ratio = excess_return / abs(avg_drawdown)

    
    return annual_volatility, sharpe_ratio, calmar_ratio, sterling_ratio





def plot_capital_allocation_over_time(df_path):
    """
    Loads the capital history from 'capital_tracking.csv' and plots 
    cash, invested and total portfolio value over time - scaled in thousands ($).
    """

    capital_df = pd.read_csv(df_path, parse_dates=["Date"])

    plt.figure(figsize=(12, 6))
    plt.plot(capital_df["Date"], capital_df["Cash"] / 1000, label="Cash", linewidth=2, color="#47A9E1")
    plt.plot(capital_df["Date"], capital_df["Invested"] / 1000, label="Invested", linewidth=2, color="#9B1D20")
    plt.plot(capital_df["Date"], capital_df["Total"] / 1000, label="Total Portfolio", linestyle="--", linewidth=2, color="#00226B")

    # Title and axis labels
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Capital (in thousand $)", fontsize=16)

    ax = plt.gca()
    ax.set_ylim(bottom=0)
    ax.set_xlim([capital_df["Date"].min(), capital_df["Date"].max()])
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Fix date formatting on x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Legend and grid
    plt.legend(fontsize=14, frameon=True, loc="best")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()





# =============================================================================
# MAIN SCRIPT
# This block executes the full process: 
# 1) Load and prepare data,
# 2) Run statistical tests to identify tradable pairs,
# 3) Simulate the trading strategy,
# 4) Evaluate and visualize performance.
# =============================================================================


def main():
    # Choose time series path
    filename = "EuroStoxx600_Weekdays.xlsx"

    # Choose parameters
    train_window = 518
    initial_capital = 100000
    z_score_threshold = 2
    min_cash_threshold = 10000
    transaction_cost_rate = 0.003
    short_fee_rate = 0.01
    risk_free_rate_annual = 0.0431
    threshold = 0.9

    price_df, log_df = load_data(filename)




    results_path = get_local_filepath("cointegration_results.csv")

    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        print("Loading existing cointegration pairs...")
        cointegration_results = pd.read_csv(results_path)
    else:
        print("Computing new cointegration pairs...")
        high_corr_pairs = find_high_corr_pairs(log_df, threshold=0.9)
        I0, I1, invalid = classify_series(log_df)
        cointegration_results = run_engle_granger_test(log_df, high_corr_pairs, I1)
        cointegration_results.to_csv(results_path, index=False)

    # Start trading simulation
    df_trades, portfolio_series, pair_signals = run_trading_simulation(
        log_df=log_df,
        price_df=price_df,
        cointegration_results = cointegration_results,
        train_window = train_window,
        initial_capital = initial_capital,
        z_score_threshold = z_score_threshold,
        min_cash_threshold = min_cash_threshold,
        transaction_cost_rate = transaction_cost_rate,
        short_fee_rate = short_fee_rate
    )

    avg_drawdown, max_drawdown = calculate_drawdown(portfolio_series)


    annual_volatility, sharpe_ratio, calmar_ratio, sterling_ratio = calculate_sharpe_ratio(
        risk_free_rate_annual = risk_free_rate_annual, 
        portfolio_series = portfolio_series, 
        max_drawdown = max_drawdown, avg_drawdown = avg_drawdown
        )
    

    

    print(f"Number of executed trades: {len(df_trades)}")

    # Optional plotting functions 
    #plot_drawdown(portfolio_series)
    #plot_portfolio_value(portfolio_series)
    #plot_daily_log_returns(portfolio_series, bins=25)
    #plot_daily_returns(portfolio_series, bins=50)



    # Save portfolio series to CSV
    capital_tracking_path = get_local_filepath("capital_tracking.csv")
    plot_capital_allocation_over_time(capital_tracking_path)

    portfolio_series.to_csv(get_local_filepath("portfolio_series.csv"), index_label="Date")

    pair_counts = df_trades["Pair"].value_counts()

    # Analyze transaction costs
    total_transaction_costs, total_short_fees, total_pnl = analyze_costs(df_trades, transaction_cost_rate=transaction_cost_rate, initial_capital=initial_capital)

    high_corr_pairs = find_high_corr_pairs(log_df, threshold=threshold)





    # Display simulation results
    print("\nSimulation results:")
    print(f"Final portfolio value: {portfolio_series.iloc[-1]:,.2f} $")
    total_return = (portfolio_series.iloc[-1] / initial_capital - 1)
    print(f"Total return: {total_return * 100:.2f}%")
    max_drawdown = (portfolio_series / portfolio_series.cummax() - 1).min()
    print(f"Annualized Volatility: {annual_volatility:.4f} ({annual_volatility*100:.2f} %)")
    print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")

    print(f"Maximum drawdown: {max_drawdown * 100:.2f}%")
    
    print(f"Average drawdown: {avg_drawdown * 100:.2f}%")
    print(f"Calmar ratio: {calmar_ratio:.2f}")
    print(f"Sterling ratio: {sterling_ratio:.2f}")

    print("Number of cointegrated pairs:", len(cointegration_results))
    print(f"Number of highly correlated pairs (>= 0.9): {len(high_corr_pairs)}")



    # === Cost Output ===
    print("\n Cost Analysis:")
    print(f"Total transaction costs:       {total_transaction_costs:,.2f} $ ({100 * total_transaction_costs / initial_capital:.2f} % of initial capital)")
    print(f"Total short selling fees:      {total_short_fees:,.2f} $ ({100 * total_short_fees / initial_capital:.2f} % of initial capital)")
    print(f"Total costs:                   {total_transaction_costs + total_short_fees:,.2f} $")
    print(f"• Costs relative to total profit: {(100 * (total_transaction_costs + total_short_fees) / abs(total_pnl)):.2f} %")
    plt.show()

    return df_trades, portfolio_series, pair_signals


if __name__ == "__main__":
    main()