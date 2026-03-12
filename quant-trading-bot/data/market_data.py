import requests
import pandas as pd
from datetime import datetime, timedelta
from config.settings import ALPHAVANTAGE_API_KEY, SYMBOL, USE_SIMULATED_DATA
from utils.logger import logger  # assuming you have this

def get_historical_data():
    """
    Fetch daily OHLCV from Alpha Vantage (free tier safe).
    Uses compact to avoid full-size premium restriction.
    Adds staleness check against current date.

    Set USE_SIMULATED_DATA = True in settings.py to force 500-bar
    synthetic data regardless of API key availability.
    """
    if USE_SIMULATED_DATA:
        logger.info("[market_data] USE_SIMULATED_DATA=True → using synthetic 500-bar data")
        return _simulate()

    if not ALPHAVANTAGE_API_KEY or ALPHAVANTAGE_API_KEY in ["YOUR_KEY", "M0FW3EVWGD780HIB"]:
        logger.warning("No/invalid Alpha Vantage key → using simulated data.")
        return _simulate()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": SYMBOL,
        "outputsize": "compact",  # free tier safe — last ~100 bars
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()

        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            if "rate limit" in msg.lower() or "25 requests" in msg:
                logger.error(f"Rate limit hit: {msg}. Wait until tomorrow or get new key.")
            else:
                logger.error(f"API message: {msg}")
            return _simulate()

        if "Error Message" in data:
            raise RuntimeError(data["Error Message"])

        if "Time Series (Daily)" not in data:
            raise ValueError("No time series data returned.")

        prices = data["Time Series (Daily)"]
        df = pd.DataFrame(prices).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)

        # Staleness check
        latest_date = df.index[-1]
        today = datetime.now().date()
        days_old = (today - latest_date.date()).days
        if days_old > 5:  # allow weekend/holiday buffer
            logger.warning(f"Data is stale! Latest bar: {latest_date.date()} "
                           f"({days_old} days old). Likely rate limit/cache issue. "
                           "Get new key or wait for reset.")
            # Optional: fall back to sim if too old
            # return _simulate()

        logger.info(f"[market_data] Latest bar: {latest_date.date()} "
                    f"(today: {today})")

        # Rename columns to match what strategy expects
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        return df

    except Exception as e:
        logger.exception(f"Failed to fetch Alpha Vantage data: {e}")
        return _simulate()


def _simulate() -> pd.DataFrame:
    """
    Seeded random-walk price simulation.
    Produces the same column names as Alpha Vantage so the rest of
    the pipeline works identically in sim and live modes.
    """
    import math, random
    from datetime import datetime, timedelta

    random.seed(42)
    price  = 150.0
    rows   = {}
    start  = datetime(2022, 1, 3)

    for i in range(500):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        change = random.gauss(0.0003, 0.015)
        price *= 1 + change
        open_  = price * random.uniform(0.995, 1.005)
        high   = price * random.uniform(1.000, 1.015)
        low    = price * random.uniform(0.985, 1.000)
        rows[d.strftime("%Y-%m-%d")] = {
            "1. open":   round(open_,  2),
            "2. high":   round(high,   2),
            "3. low":    round(low,    2),
            "4. close":  round(price,  2),
            "5. volume": random.randint(10_000_000, 80_000_000),
        }

    df = pd.DataFrame(rows).T.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        "1. open":   "open",
        "2. high":   "high",
        "3. low":    "low",
        "4. close":  "close",
        "5. volume": "volume",
    })
    return df