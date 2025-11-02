import os
import argparse
import yaml
import pandas as pd
import numpy as np
from glob import glob
import os.path as osp

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def build_price_matrix(data_dir, universe=None):
    dfs = {}
    files = glob(osp.join(data_dir, "*.csv"))
    if universe:
        files = [osp.join(data_dir, f"{s}.csv") for s in universe]
    for path in files:
        if not osp.exists(path):
            continue
        sym = osp.splitext(osp.basename(path))[0]
        d = pd.read_csv(path, parse_dates=[0], index_col=0)
        if 'close' not in d.columns:
            # assume single-column CSV of closes
            col = d.columns[0]
            d = d.rename(columns={col: 'close'})
        series = d['close'].sort_index()
        dfs[sym] = series
    price = pd.DataFrame(dfs).sort_index()
    # drop columns all-NaN and rows with too many NaNs
    price = price.dropna(axis=1, how='all')
    price = price.dropna(axis=0, thresh=max(1, int(0.5*price.shape[1])))
    # forward/backfill small holes
    price = price.fillna(method='ffill').fillna(method='bfill')
    return price

def estimate_adv_placeholder(price, window=4):
    """
    Placeholder ADV estimator: since we only have weekly closes (no volume),
    approximate ADV via weekly absolute returns scale. If you later fetch volume
    from Public client, replace this with real ADV.
    """
    rets = price.pct_change().abs()
    adv_proxy = rets.rolling(window=window, min_periods=1).mean()
    scale = 1e6
    adv_est = adv_proxy.fillna(method='bfill').iloc[-1] * scale
    adv_est = adv_est.replace(0, 1.0)
    return adv_est

def compute_returns(price):
    rets = price.pct_change().dropna()
    return rets

def save_processed(price, rets, adv, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    price.to_csv(osp.join(data_dir, "processed_prices.csv"))
    rets.to_csv(osp.join(data_dir, "processed_rets.csv"))
    adv.to_frame('adv').to_csv(osp.join(data_dir, "adv_estimates.csv"))

def main(config_path):
    cfg = load_config(config_path)
    data_dir = cfg.get('data_cache', 'data/')
    processed_dir = cfg.get('processed_dir', data_dir)
    universe = cfg.get('universe', None)
    price = build_price_matrix(data_dir, universe)
    rets = compute_returns(price)
    adv = estimate_adv_placeholder(price)
    save_processed(price, rets, adv, data_dir)
    print(f"Saved processed files to {data_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)
