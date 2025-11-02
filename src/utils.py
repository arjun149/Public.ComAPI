import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as osp

sns.set(style="darkgrid", rc={"figure.figsize": (10, 6)})

def ann_return(returns, periods_per_year=52):
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1

def ann_vol(returns, periods_per_year=52):
    returns = np.array(returns)
    if len(returns) <= 1:
        return 0.0
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(returns, periods_per_year=52, rf=0.0):
    ar = ann_return(returns, periods_per_year)
    av = ann_vol(returns, periods_per_year)
    if av == 0:
        return np.nan
    return (ar - rf) / av

def max_drawdown(returns):
    cum = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min()

def plot_equity_curve(returns, outpath=None, label='Portfolio'):
    cum = pd.Series(np.cumprod(1 + pd.Series(returns)))
    ax = cum.plot(title=f'Equity Curve: {label}', ylabel='Cumulative Return')
    if outpath:
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        ax.get_figure().savefig(outpath)
    plt.close()

def plot_rolling_sharpe(returns, window=26, outpath=None):
    s = pd.Series(returns)
    roll_sharpe = s.rolling(window).mean() / s.rolling(window).std() * np.sqrt(52)
    ax = roll_sharpe.plot(title='Rolling Sharpe', ylabel='Sharpe')
    if outpath:
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        ax.get_figure().savefig(outpath)
    plt.close()

def plot_weight_heatmap(weights, dates, assets, outpath=None):
    df = pd.DataFrame(weights, index=dates[:weights.shape[0]], columns=assets)
    plt.figure(figsize=(12, max(4, len(assets)/3)))
    sns.heatmap(df.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'weight'})
    plt.xlabel('Date')
    plt.ylabel('Asset')
    plt.title('Weights Heatmap')
    if outpath:
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, bbox_inches='tight')
    plt.close()
