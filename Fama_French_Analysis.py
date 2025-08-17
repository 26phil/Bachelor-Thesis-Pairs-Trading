"""
===============================================================================
 Author         : Philipp König
 Final Version  : 2025-08-14
===============================================================================

Description:
------------
Runs a Fama–French three-factor regression on a portfolio's excess returns 
relative to the risk-free rate. The script:

1. **Loads Fama–French data**:
   - Reads factor returns (Mkt-RF, SMB, HML) and risk-free rate (RF)
   - Reads portfolio returns

2. **Prepares excess returns**:
   - Calculates portfolio excess return over RF

3. **Runs regression**:
   - Uses OLS to regress portfolio excess returns on Mkt-RF, SMB, HML
   - Adds significance stars based on p-values

4. **Outputs results**:
   - Coefficients and t-statistics printed to console
   - Full `statsmodels` regression summary

Usage:
------
Fama_French_Analysis.py

Dependencies:
-------------
- Python 3.12
- pandas
- statsmodels

Input Files:
------------
- Fama_French_Data.xlsx : Must contain columns:
    'Date', 'Mkt-RF', 'SMB', 'HML', 'RF', and 'Portfolio Value'

Notes:
------
- The Fama–French factor data should be aligned with the portfolio's return frequency
- Coefficient stars: *** p<0.01, ** p<0.05, * p<0.1
"""




import pandas as pd
import statsmodels.api as sm

# === 1. Load Data ===
df = pd.read_excel('Fama_French_Data.xlsx', parse_dates=['Date'], index_col='Date')

# === 2. Extract Columns ===
mkt_rf = df['Mkt-RF']
smb = df['SMB']
hml = df['HML']
rf = df['RF']

# === 3. Portfolio Return ===
df['pf_return'] = df['Portfolio Value']

# Fama-French RF is in percent
df['pf_excess_return'] = df['pf_return'] - rf

# === 4. Regression ===
df_reg = df.dropna(subset=['pf_excess_return', 'Mkt-RF', 'SMB', 'HML'])
Y = df_reg['pf_excess_return']
X = sm.add_constant(df_reg[['Mkt-RF', 'SMB', 'HML']]) 

model = sm.OLS(Y, X).fit()

# === 5. Output ===
print("\nRegression Coefficients with t-Statistics:")
for param_name, coef, t_stat in zip(model.params.index, model.params, model.tvalues):
    print(f"{param_name}: {coef:.5f}  (t = {t_stat:.2f})")

print("\nLaTeX Table:")
print(r"\begin{tabular}{lcc}")
print(r"\hline")
print(r" & Coefficient & (t-Statistic) \\")
print(r"\hline")
for param_name, coef, t_stat in zip(model.params.index, model.params, model.tvalues):
    coef_stars = ''
    if model.pvalues[param_name] < 0.01:
        coef_stars = '***'
    elif model.pvalues[param_name] < 0.05:
        coef_stars = '**'
    elif model.pvalues[param_name] < 0.1:
        coef_stars = '*'
    print(f"{param_name} & {coef:.5f}{coef_stars} & ({t_stat:.2f}) \\\\")
print(r"\hline")
print(r"\end{tabular}")

print(model.summary())

