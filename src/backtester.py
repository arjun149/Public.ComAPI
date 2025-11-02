import argparse
import yaml
import os
import pandas as pd
import numpy as np
from glob import glob
import os.path as osp
from src.optimizers import equal_weight, min_variance, mean_variance
from src import utils

def load_processed(data_dir):
    price = pd.read_csv(osp.join(data_dir, "processed_prices.csv"), index_col=0, parse_dates=[0])
    rets = pd.read_csv(osp.join(data_dir, "processed_rets.csv"), index_col=0, parse_dates=[0])
    adv = pd.read_csv(osp.join(data_dir, "adv_estimates.csv"), index_col=0)
    adv = adv['adv']
    return price, rets, adv

def backtest(cfg, methods):
    data_dir = cfg.get('data_cache', 'data/')
    price, rets, adv = load_processed(data_dir)
    dates = rets.index
    assets = rets.columns.tolist()
    results = {}
    for m in methods:
        w_hist = []
        port_ret = []
        prev_w = np.zeros(len(assets))
        for t in range(len(dates)-1):
            # compute estimates using expanding lookback
            lookback_start = max(0, t-52)
            window = rets.iloc[lookback_start:t+1]
            if window.shape[0] < 2:
                cov = rets.iloc[:t+1].cov().values
                mu = rets.iloc[:t+1].mean().values
            else:
                cov = window.cov().values
                mu = window.mean().values
            if m == 'equal':
                w = equal_weight(len(mu))
            elif m == 'minvar':
                w = min_variance(cov)
            elif m == 'meanvar':
                w = mean_variance(mu, cov, risk_aversion=1.0)
            else:
                raise ValueError(m)
            # scale to target vol
            port_vol = np.sqrt(w.T @ cov @ w) * np.sqrt(52)
            target_vol = cfg.get('target_vol', 0.10)
            scale = target_vol / (port_vol + 1e-9)
            w = w * scale
            # box constraints per-asset
            max_pos = cfg.get('per_asset_max', None)
            if max_pos:
                w = np.clip(w, -max_pos, max_pos)
            # enforce max gross leverage
            max_gross = cfg.get('max_gross_leverage', 2.0)
            gross = np.sum(np.abs(w))
            if gross > max_gross:
                w = w * (max_gross / gross)
                gross = np.sum(np.abs(w))
            # compute next-period realized return
            r_next = rets.iloc[t+1].values
            port_r = np.dot(w, r_next)
            # transaction cost and slippage
            turnover = np.sum(np.abs(w - prev_w))
            tc_bps = cfg.get('transaction_cost_bps', 10) / 10000.0
            tc = turnover * tc_bps
            # ADV-based slippage: for each asset, slippage = base_slippage * (trade_size / ADV)
            base_slip = cfg.get('base_slippage_pct', 0.001)  # default 0.1%
            trade_sizes = np.abs(w - prev_w)  # fractional weights
            # map fractional trade to USD using notional = portfolio_value * abs(weight)
            notional = cfg.get('start_capital', 100000)
            trade_usd = trade_sizes * notional
            adv_vals = adv.reindex(assets).fillna(1e6).values  # USD adv estimate
            slip_per_asset = base_slip * (trade_usd / (adv_vals + 1e-9))
            slip = np.sum(slip_per_asset)
            net_r = port_r - tc - slip
            # margin interest for gross leverage > 1.0
            margin_rate = cfg.get('margin_rate_annual', 0.03)  # 3% per annum default
            if gross > 1.0:
                # interest approximated per period
                interest = (gross - 1.0) * margin_rate / 52.0
            else:
                interest = 0.0
            net_r = net_r - interest
            port_ret.append(net_r)
            w_hist.append(w)
            prev_w = w
        # metrics
        port_ret = np.array(port_ret)
        ann_r = utils.ann_return(port_ret)
        ann_v = utils.ann_vol(port_ret)
        sr = utils.sharpe_ratio(port_ret)
        mdd = utils.max_drawdown(port_ret)
        results[m] = {'returns': port_ret, 'weights': np.vstack(w_hist) if w_hist else np.zeros((0,len(assets))),
                      'sharpe': sr, 'ann_return': ann_r, 'ann_vol': ann_v, 'max_dd': mdd,
                      'dates': dates[1:1+len(port_ret)], 'assets': assets}
        # persist per-method outputs
        out_dir = cfg.get('results_dir', 'results/')
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(results[m]['weights'], index=results[m]['dates'], columns=assets).to_csv(osp.join(out_dir, f"weights_{m}.csv"))
        pd.DataFrame({'returns': results[m]['returns']}, index=results[m]['dates']).to_csv(osp.join(out_dir, f"returns_{m}.csv"))
        # plots
        utils.plot_equity_curve(results[m]['returns'], outpath=osp.join(out_dir, f"equity_{m}.png"), label=m)
        utils.plot_rolling_sharpe(results[m]['returns'], outpath=osp.join(out_dir, f"rolling_sharpe_{m}.png"))
        utils.plot_weight_heatmap(results[m]['weights'], results[m]['dates'], assets, outpath=osp.join(out_dir, f"weights_heatmap_{m}.png"))
    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--methods', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    methods = args.methods.split(',')
    res = backtest(cfg, methods)
    for m,v in res.items():
        print(m, 'sharpe:', v['sharpe'], 'ann_return:', v['ann_return'], 'ann_vol:', v['ann_vol'], 'max_dd:', v['max_dd'])
