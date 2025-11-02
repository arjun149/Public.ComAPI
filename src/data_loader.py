import os
import argparse
import yaml
import pandas as pd
import os.path as osp

# Try import of official client; adjust package name if different
try:
    import publicdotcom
    from publicdotcom import Public
except Exception:
    Public = None

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def init_client(api_key_env="PUBLICDOTCOM_API_KEY"):
    """
    Initialize the Public client using environment variable.
    If your client requires different credentials change this function.
    """
    api_key = os.getenv(api_key_env)
    if Public is None:
        raise RuntimeError("publicdotcom client package not installed. pip install publicdotcom")
    if not api_key:
        raise RuntimeError(f"PUBLICDOTCOM API key not found in env var {api_key_env}")
    # Example: Public(api_key=api_key) — adapt if the real client uses a different signature
    client = Public(api_key=api_key)
    return client

def fetch_symbol_daily(client, symbol, start, end):
    """
    Fetch daily OHLCV using publicdotcom client and return a DataFrame with Date index and 'close' column.
    NOTE: adapt field access if the client's method signatures differ.
    """
    tried = []
    resp = None
    # Attempt 1
    try:
        resp = client.get_historical_prices(symbol=symbol, start=start, end=end, interval="1d")
    except Exception as e:
        tried.append(("get_historical_prices", str(e)))
    if resp is None:
        try:
            resp = client.market.get_historical_prices(symbol=symbol, start=start, end=end, interval="1d")
        except Exception as e:
            tried.append(("market.get_historical_prices", str(e)))
    if resp is None:
        try:
            resp = client.get_prices(symbol=symbol, start=start, end=end, interval="1d")
        except Exception as e:
            tried.append(("get_prices", str(e)))
    if resp is None:
        try:
            resp = client.historical_prices(symbol=symbol, start=start, end=end, interval="1d")
        except Exception as e:
            tried.append(("historical_prices", str(e)))
    if resp is None:
        raise RuntimeError(f"Could not fetch prices for {symbol}; tried: {tried}")

    # Normalize response to DataFrame
    if isinstance(resp, pd.DataFrame):
        df = resp.copy()
    else:
        try:
            df = pd.DataFrame(resp)
        except Exception:
            raise RuntimeError("Unexpected response format from publicdotcom client for symbol " + symbol)
    # Normalize columns. Expecting timestamp/date and close present.
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    elif "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise RuntimeError("No timestamp/date column in response for " + symbol)
    # prefer 'close' column; adapt if 'close_price' or 'adj_close' present
    close_col = None
    for c in ("close", "close_price", "price", "adj_close"):
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise RuntimeError("No close-like column in response for " + symbol + "; columns: " + ", ".join(df.columns))
    df.set_index("date", inplace=True)
    series = df[close_col].sort_index()
    # Resample to weekly close (Friday); choose W-FRI to align with Friday close
    weekly = series.resample("W-FRI").last().dropna()
    weekly = weekly.to_frame(name="close")
    return weekly

def cache_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)

def main(config_path):
    cfg = load_config(config_path)
    cache_dir = cfg.get("data_cache", "data/")
    start = cfg.get("start_date")
    end = cfg.get("end_date")
    universe = cfg.get("universe", [])

    if not universe:
        universe = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "MATIC", "LTC"]

    client = init_client()

    for sym in universe:
        try:
            df = fetch_symbol_daily(client, sym, start, end)
        except Exception as e:
            print(f"Warning: failed to fetch {sym}: {e}")
            continue
        cache_path = osp.join(cache_dir, f"{sym}.csv")
        cache_df(df, cache_path)
        print(f"Cached {sym} -> {cache_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)
