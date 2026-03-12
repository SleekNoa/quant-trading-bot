"""
Multi-Ticker Data Fetcher
==========================
Fetches and caches historical OHLCV data for multiple tickers while
respecting API rate limits (Alpha Vantage: ≤ 5 req/min on free tier,
yfinance: no hard limit but polite delay is good practice).

Architecture
------------
    fetch_all_tickers(tickers, ...)  →  dict[str, pd.DataFrame]
    fetch_ticker_safe(ticker, ...)   →  pd.DataFrame | None    (single)

The fetcher delegates to the project's existing get_historical_data()
function.  It passes the ticker as a ``symbol`` keyword argument so it
works with both:
    • The Alpha Vantage-based market_data.py  (uses SYMBOL config)
    • The yfinance-based market_data.py       (accepts symbol kwarg)

Rate limiting
-------------
    delay_sec = 13.0   →  ≤4.6 requests/min  (well within the 5/min cap)
    The delay is injected between calls, not before the first call.

Usage
-----
    from data.multi_ticker import fetch_all_tickers
    data = fetch_all_tickers(["AAPL", "MSFT", "SPY"])
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} bars")
"""

from __future__ import annotations

import time
import pandas as pd
from typing import Optional

from utils.logger import logger


# ── Single-ticker safe fetch ───────────────────────────────────────────────────

def fetch_ticker_safe(
    ticker:    str,
    min_bars:  int = 50,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for one ticker.

    Relies on get_historical_data(symbol=...) which now supports passing the ticker.
    Returns None if fetch fails or too few bars.
    """
    from data.market_data import get_historical_data

    try:
        df = get_historical_data(symbol=ticker)

        if df is None or len(df) < min_bars:
            bar_count = len(df) if df is not None else 0
            logger.warning(
                f"           {ticker}: skipped — only {bar_count} bars "
                f"(need ≥ {min_bars})"
            )
            return None

        return df

    except Exception as exc:
        logger.error(f"           {ticker}: fetch failed — {exc}")
        return None

    return df


# ── Multi-ticker batch fetch ───────────────────────────────────────────────────

def fetch_all_tickers(
    tickers:   list[str],
    delay_sec: float = 13.0,
    min_bars:  int   = 50,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for every ticker in the list.

    Parameters
    ----------
    tickers   : list of symbol strings, e.g. ["SPY", "QQQ", "AAPL"]
    delay_sec : pause between successive API calls (seconds).
                Default 13 s keeps Alpha Vantage well under 5 req/min.
    min_bars  : minimum bars required to include a ticker in the result

    Returns
    -------
    dict  { ticker: DataFrame }  — only successfully loaded tickers included

    Notes
    -----
    Progress is logged via utils.logger at INFO level.
    Failures are logged as ERROR but do not raise — the caller receives
    whatever tickers loaded successfully.
    """
    result: dict[str, pd.DataFrame] = {}
    total = len(tickers)

    logger.info(f"           Multi-ticker scan: {total} ticker(s) — delay {delay_sec}s between calls")

    for i, ticker in enumerate(tickers):
        logger.info(f"           [{i + 1}/{total}] Fetching {ticker}...")

        df = fetch_ticker_safe(ticker, min_bars=min_bars)

        if df is not None:
            result[ticker] = df
            start_dt = df.index[0].strftime("%Y-%m-%d") if hasattr(df.index[0], "strftime") else str(df.index[0])
            end_dt   = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])
            logger.info(f"           {ticker}: {len(df)} bars  ({start_dt} → {end_dt})")

        # Rate-limit delay between calls (not after the last one)
        if i < total - 1:
            time.sleep(delay_sec)

    n_ok = len(result)
    n_fail = total - n_ok
    logger.info(
        f"           Multi-ticker complete: "
        f"{n_ok} loaded, {n_fail} skipped/failed"
    )

    return result