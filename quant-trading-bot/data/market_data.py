"""
data/market_data.py — NOW USING yfinance (unlimited, no key, always fresh)
=====================================================================
Replaced Alpha Vantage (rate-limited + stale 2023 data) with yfinance.
Zero setup. Runs every day with real 2026 data.
"""

# === KEPT (still used) ===
import pandas as pd
from datetime import datetime, timedelta
from config.settings import SYMBOL, USE_SIMULATED_DATA
from utils.logger import logger

# === REMOVED / COMMENTED OUT (Alpha Vantage no longer needed) ===
# import requests
# from config.settings import ALPHAVANTAGE_API_KEY   # ← no longer used here

# === NEW: yfinance (this is the only thing you actually need) ===
import yfinance as yf


# ... (keep all existing imports)

def get_historical_data(symbol: str = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV.
    - If symbol is None → uses SYMBOL from config.settings
    - Otherwise → uses the provided symbol
    - USE_SIMULATED_DATA = True → forces 500-bar synthetic data (ignores symbol)
    """
    if USE_SIMULATED_DATA:
        logger.info("[market_data] USE_SIMULATED_DATA=True → using synthetic 500-bar data")
        return _simulate()   # simulation doesn't use symbol anyway

    # Determine which symbol to fetch
    fetch_symbol = symbol if symbol is not None else SYMBOL

    try:
        end = datetime.now()
        start = end - timedelta(days=5*365 + 100)  # ~5 years + buffer

        df = yf.download(fetch_symbol, start=start, end=end, progress=False, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data returned for {fetch_symbol}")

        # Standardize column names (your strategies expect lowercase)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.name = 'date'

        logger.info(f"✅ yfinance: {len(df):,} fresh bars loaded for {fetch_symbol} "
                    f"({df.index[0].date()} → {df.index[-1].date()})")
        df.attrs["simulated"] = False
        return df

    except Exception as e:
        logger.warning(f"yfinance failed for {fetch_symbol} ({e}) — falling back to simulation")
        return _simulate()

# === KEPT EXACTLY AS YOU HAD IT (fallback simulation) ===
def _simulate() -> pd.DataFrame:
    """Seeded random-walk price simulation. Produces the same column names as before."""
    import math, random
    from datetime import datetime, timedelta

    random.seed(42)
    price = 150.0
    rows = {}
    start = datetime(2022, 1, 3)

    for i in range(500):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        change = random.gauss(0.0003, 0.015)
        price *= 1 + change
        open_ = price * random.uniform(0.995, 1.005)
        high = price * random.uniform(1.000, 1.015)
        low = price * random.uniform(0.985, 1.000)
        rows[d.strftime("%Y-%m-%d")] = {
            "1. open": round(open_, 2),
            "2. high": round(high, 2),
            "3. low": round(low, 2),
            "4. close": round(price, 2),
            "5. volume": random.randint(10_000_000, 80_000_000),
        }

    df = pd.DataFrame(rows).T.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low",
        "4. close": "close", "5. volume": "volume"
    })
    df.attrs["simulated"] = True
    return df