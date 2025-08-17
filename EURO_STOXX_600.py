"""
===============================================================================
 Author         : Philipp König
 Final Version  : 2025-08-14
===============================================================================

Description:
------------
Compares the performance of a Buy-and-Hold investment in the EURO STOXX 600 
with a Pairs Trading strategy portfolio. The script performs:

1. **Data Loading & Preparation**:
   - Reads EURO STOXX 600 index prices from Excel
   - Reads Pairs Trading portfolio values from CSV
   - Calculates daily log returns and cumulative returns
   - Aligns time series for correlation and beta analysis

2. **Performance Analysis**:
   - Calculates alpha and beta via OLS regression
   - Computes risk and performance metrics:
        • Total return
        • Annualized volatility
        • Sharpe ratio
        • Maximum drawdown
        • Average drawdown
        • Calmar ratio
        • Sterling ratio
   - Calculates correlation between strategies

3. **Visualization**:
   - Portfolio value comparison chart
   - Drawdown comparison plot
   - Histogram and density of daily log returns
   - Summary statistics (mean, std, skewness, kurtosis)

4. **Statistical Outputs**:
   - Regression alpha and beta
   - Summary of performance metrics
   - Return distribution characteristics

Usage:
------
python portfolio_comparison_analysis.py

Dependencies:
-------------
- Python 3.12
- pandas
- numpy
- matplotlib
- statsmodels

Input Files:
------------
- EURO_STOXX_600_Index.xlsx : EURO STOXX 600 index prices
- portfolio_series.csv      : Pairs Trading portfolio values

Notes:
------
- The analysis starts from '2024-06-03' for both portfolios
- All returns are log returns; risk-free rate is set to 4.31% p.a.
- Plots and metrics are aligned to the overlapping date range
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
from scipy.stats import skew, kurtosis
import matplotlib.ticker as mtick

# === 1. Load EURO STOXX data ===
def load_data_euro(filename):
    data = pd.read_excel(filename, usecols="B", skiprows=1, index_col=None, header=None)
    data.columns = ["Price"]
    date_index = pd.read_excel(filename, usecols="A", skiprows=1, header=None)
    data.index = pd.to_datetime(date_index.iloc[:, 0])
    return data["Price"]

# === 2. Calculate log returns ===
def calc_log_return(series):
    series = series.replace(0, np.nan)
    series = series[series > 0]  # only positive prices
    log_return = np.log(series / series.shift(1))
    return log_return.dropna()



# === 3. Plot Buy-and-Hold vs. Pairs Trading ===
def plot_two_portfolios(buy_hold_series, pairs_series, initial_value=100000):
    plt.figure(figsize=(12, 6))
    
    # Plot lines
    plt.plot(buy_hold_series, label="STOXX Europe 600", linewidth=3, color="#00226B")
    plt.plot(pairs_series, label="Pairs Trading", linewidth=3, color="#47A9E1")
    plt.axhline(initial_value, color="gray", linestyle="--", label="Initial Value")
    
    # Title and labels
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Portfolio Value ($)", fontsize=16)
    plt.legend(loc="best", frameon=True, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Format y-axis
    formatter = FuncFormatter(lambda x, _: f'{x/1000:.0f}k')
    plt.gca().yaxis.set_major_formatter(formatter)

    # Format x-axis dates cleanly
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Limit x-axis to overlapping period
    common_index = buy_hold_series.index.intersection(pairs_series.index)
    if not common_index.empty:
        plt.xlim([common_index.min(), common_index.max()])
    
    plt.tight_layout()





# === 4. Load EURO STOXX and calculate Buy-and-Hold Portfolio ===
filename_euro = "EURO_STOXX_600_Index.xlsx"
euro_stoxx = load_data_euro(filename_euro)

log_return_euro = calc_log_return(euro_stoxx)
start_date = "2024-06-03"
log_return_euro = log_return_euro[log_return_euro.index >= start_date]

initial_value = 100000
cumulative_return_euro = np.exp(log_return_euro.cumsum())
buy_hold_portfolio = initial_value * cumulative_return_euro

# === 5. Load Pairs Trading Portfolio ===
pairs_portfolio_series = pd.read_csv("portfolio_series.csv", parse_dates=["Date"], index_col="Date")

if isinstance(pairs_portfolio_series, pd.DataFrame):
    pairs_portfolio_series = pairs_portfolio_series.iloc[:, 0]

pairs_log_returns = calc_log_return(pairs_portfolio_series)
pairs_cumulative_return = np.exp(pairs_log_returns.cumsum())
pairs_portfolio_rebased = initial_value * pairs_cumulative_return
pairs_portfolio_rebased = pairs_portfolio_rebased[pairs_portfolio_rebased.index >= start_date]

# === 6. Align log returns for correlation and beta ===
if isinstance(log_return_euro, pd.DataFrame):
    log_return_euro = log_return_euro.iloc[:, 0]

aligned_returns = pd.concat([log_return_euro, pairs_log_returns], axis=1, join='inner')
aligned_returns.columns = ["BuyHold_LogReturn", "Pairs_LogReturn"]

# === 7. Calculate Beta via OLS regression ===
X = add_constant(aligned_returns["BuyHold_LogReturn"])
y = aligned_returns["Pairs_LogReturn"]
model = OLS(y, X).fit()
beta = model.params[1]
alpha = model.params[0]

# === 8. Output correlation and beta ===
correlation = aligned_returns.corr().iloc[0, 1]
print("Log returns correlation:", round(correlation, 4))
print("Beta of Pairs Trading Portfolio vs EURO STOXX:", round(beta, 4))

# === 9. Plot the portfolios ===
plot_two_portfolios(buy_hold_portfolio, pairs_portfolio_rebased, initial_value)



# === 10. Performance metrics for Euro Stoxx 600 Buy-and-Hold ===

# 1. Total return
total_return_euro = (buy_hold_portfolio.iloc[-1] / 100000) - 1

# 2. Annualized volatility (based on log returns)
volatility_annual_euro = log_return_euro.std() * np.sqrt(252)

# 4. Sharpe ratio
risk_free_rate = 0.0431	
sharpe_ratio_euro = (total_return_euro - risk_free_rate) / volatility_annual_euro

# 5. Drawdown
rolling_max = buy_hold_portfolio.cummax()
drawdown = (buy_hold_portfolio / rolling_max) - 1
max_drawdown = drawdown.min()

def calculate_average_drawdown(series):
    drawdown = (series / series.cummax()) - 1
    in_drawdown = False
    drawdowns = []
    current_dd = []

    for dd in drawdown:
        if dd < 0:
            current_dd.append(dd)
            in_drawdown = True
        elif in_drawdown:
            drawdowns.append(np.mean(current_dd))
            current_dd = []
            in_drawdown = False

    if current_dd:  # if a drawdown is still ongoing at the end
        drawdowns.append(np.mean(current_dd))

    if drawdowns:
        return np.mean(drawdowns)
    else:
        return 0.0

average_drawdown = calculate_average_drawdown(buy_hold_portfolio)

# === 11. Output performance metrics ===
print("\nSTOXX Europe 600 600 Performance Metrics (Buy-and-Hold):")
print(f"Final portfolio value STOXX Europe 600 (Buy-and-Hold): {buy_hold_portfolio.iloc[-1]:,.2f} $")
print(f"Total return: {total_return_euro * 100:.2f} %")
print(f"Annualized volatility: {volatility_annual_euro * 100:.2f} %")
print(f"Sharpe ratio: {sharpe_ratio_euro:.2f}")
print(f"Maximum drawdown: {max_drawdown * 100:.2f} %")
print(f"Average drawdown: {average_drawdown * 100:.2f} %")

def plot_max_drawdown_comparison(buy_hold, pairs, savepath=None):


    # Drawdowns berechnen
    dd_buy_hold = (buy_hold / buy_hold.cummax()) - 1
    dd_pairs    = (pairs    / pairs.cummax())    - 1

    # Falls DataFrames: erste Spalte nehmen
    if isinstance(dd_buy_hold, pd.DataFrame):
        dd_buy_hold = dd_buy_hold.iloc[:, 0]
    if isinstance(dd_pairs, pd.DataFrame):
        dd_pairs = dd_pairs.iloc[:, 0]

    # Gemeinsamen Index ausrichten
    common_index = dd_buy_hold.index.intersection(dd_pairs.index)
    dd_buy_hold = dd_buy_hold.loc[common_index]
    dd_pairs    = dd_pairs.loc[common_index]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dd_buy_hold, label="Buy & Hold Drawdown", linewidth=2, color="#47A9E1")
    plt.plot(dd_pairs,    label="Pairs Trading Drawdown", linewidth=2, color="#00226B")

    # Flächen füllen
    plt.fill_between(dd_buy_hold.index, dd_buy_hold, 0, color="#47A9E1", alpha=0.2)
    plt.fill_between(dd_pairs.index,    dd_pairs,    0, color="#00226B", alpha=0.2)

    # Achsen & Formatierung
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Drawdown (%)", fontsize=16)
    # Prozent ohne Minuszeichen anzeigen (absoluter Wert)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{abs(y*100):.0f}"))

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Datumsformat
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # X-Achse begrenzen
    plt.xlim([common_index.min(), common_index.max()])

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")



# === 12. Calmar and Sterling Ratio ===

# 1. Calmar Ratio = Annualized Return / Max Drawdown
calmar_ratio = (total_return_euro - risk_free_rate) / abs(max_drawdown)

# 2. Sterling Ratio = Annualized Return / Average Drawdown
sterling_ratio = (total_return_euro - risk_free_rate) / abs(average_drawdown)

# === Print Results ===
print(f"Calmar ratio: {calmar_ratio:.2f}")
print(f"Sterling ratio: {sterling_ratio:.2f}")



def plot_log_return_histogram_with_density(buy_hold_series, pairs_series, bins=50):
    log_return_euro = np.log(buy_hold_series / buy_hold_series.shift(1)).dropna()
    log_return_pairs = np.log(pairs_series / pairs_series.shift(1)).dropna()

    combined_returns = pd.concat([log_return_euro, log_return_pairs])
    bin_edges = np.histogram_bin_edges(combined_returns, bins=bins)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    ax.hist(log_return_euro, bins=bin_edges, color="#47A9E1", edgecolor='black',
            alpha=0.5, density=True, label="STOXX Europe 600 Histogram")
    ax.hist(log_return_pairs, bins=bin_edges, color="#00226B", edgecolor='black',
            alpha=0.5, density=True, label="Pairs Trading Histogram")

    # Title
    ax.set_xlabel("Daily Log Return", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)

    # Tick-Formatierung
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Legend & Grid
    ax.legend(frameon=True, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

 


def print_return_statistics(series, name="Series"):
    log_returns = np.log(series / series.shift(1)).dropna()
    print(f"\n{name} Statistics:")
    print(f"Mean: {log_returns.mean():.5f}")
    print(f"Std Dev: {log_returns.std():.5f}")
    print(f"Skewness: {skew(log_returns):.2f}")
    print(f"Kurtosis: {kurtosis(log_returns):.2f}")




# Call the function
plot_max_drawdown_comparison(buy_hold_portfolio, pairs_portfolio_rebased)
plot_log_return_histogram_with_density(buy_hold_portfolio, pairs_portfolio_rebased)
print_return_statistics(buy_hold_portfolio, "STOXX Europe 600")
print_return_statistics(pairs_portfolio_rebased, "Pairs Trading Portfolio")
print(f"Alpha of Pairs Trading Portfolio vs EURO STOXX: {alpha:.5f}")