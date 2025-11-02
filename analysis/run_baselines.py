import argparse
import yaml
import os
import pandas as pd
import sys
sys.path.append('.')  # ensure src importable
from src import preprocess
from src import backtester

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    # preprocess
    print("Preprocessing price data...")
    preprocess.main(config_path)
    print("Running backtests...")
    methods = cfg.get('methods', ['equal','minvar','meanvar'])
    res = backtester.backtest(cfg, methods)
    # summarize
    rows = []
    for m,v in res.items():
        rows.append({'method': m, 'sharpe': v['sharpe'], 'ann_return': v['ann_return'], 'ann_vol': v['ann_vol'], 'max_dd': v['max_dd']})
    outdir = cfg.get('results_dir', 'results/')
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'summary_metrics.csv'), index=False)
    print("Saved summary to", os.path.join(outdir, 'summary_metrics.csv'))
    print("Done.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)
