"""
data/multi_ticker.py — Multi-Ticker Data Fetcher
==================================================
Fetches historical OHLCV data for multiple tickers while respecting
API rate limits.

Simulated / stale data is rejected: any ticker whose last bar is older
than STALENESS_THRESHOLD_DAYS is treated as a failed fetch and excluded
from results. This prevents synthetic fallback data from contaminating
signal rankings.
"""

from __future__ import annotations

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from utils.logger import logger

# Simulated fallback data always ends 2023-05-17 (~1000 days ago).
# Any ticker whose last bar is older than this threshold is rejected.
_STALENESS_DAYS = 90


# ── Single-ticker safe fetch ───────────────────────────────────────────────────

def fetch_ticker_safe(
    ticker:   str,
    min_bars: int = 50,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data for one ticker.
    Returns None if fetch fails, too few bars, or data is stale/simulated.
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

        # Reject simulated/stale data — last bar must be within 90 days
        last_date = pd.Timestamp(df.index[-1]).to_pydatetime().replace(tzinfo=None)
        cutoff    = datetime.now() - timedelta(days=_STALENESS_DAYS)
        if last_date < cutoff:
            logger.warning(
                f"           {ticker}: skipped — last bar {last_date.date()} "
                f"is stale (simulated fallback or delisted)"
            )
            return None

        return df

    except Exception as exc:
        logger.error(f"           {ticker}: fetch failed — {exc}")
        return None


# ── Multi-ticker batch fetch ───────────────────────────────────────────────────

def fetch_all_tickers(
    tickers:   list[str],
    delay_sec: float = 13.0,
    min_bars:  int   = 50,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for every ticker in the list.
    Only tickers with real, recent data are included in the result.

    Parameters
    ----------
    tickers   : list of symbol strings
    delay_sec : pause between successive API calls (seconds)
    min_bars  : minimum bars required to include a ticker

    Returns
    -------
    dict { ticker: DataFrame } — successfully loaded tickers only
    """
    result: dict[str, pd.DataFrame] = {}
    total  = len(tickers)
    n_fail = 0

    logger.info(
        f"           Multi-ticker scan: {total} ticker(s) "
        f"— delay {delay_sec}s between calls"
    )

    for i, ticker in enumerate(tickers):
        logger.info(f"           [{i + 1}/{total}] Fetching {ticker}...")

        df = fetch_ticker_safe(ticker, min_bars=min_bars)

        if df is not None:
            result[ticker] = df
            start_dt = df.index[0].strftime("%Y-%m-%d")
            end_dt   = df.index[-1].strftime("%Y-%m-%d")
            logger.info(f"           {ticker}: {len(df)} bars  ({start_dt} → {end_dt})")
        else:
            n_fail += 1

        if i < total - 1:
            time.sleep(delay_sec)

    n_ok = len(result)
    logger.info(
        f"           Multi-ticker complete: "
        f"{n_ok} loaded, {n_fail} skipped/failed"
    )

    return result