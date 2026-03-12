"""
strategies/directional_change_strategy.py
==========================================
Directional Changes (DC) — event-based trend detection.

Source:
    Long, Kampouridis & Papastylianou (2026).
    "Multi-objective genetic programming-based algorithmic trading,
    using directional changes and a modified sharpe ratio score."
    Artificial Intelligence Review. Springer.
    DOI: 10.1007/s10462-025-11390-9

Background:
    Unlike MACD/RSI which sample at fixed time intervals (daily, hourly),
    DC summarises price into a series of EVENTS — directional changes and
    overshoots. This filters out noise and focuses on *meaningful* moves
    exceeding threshold θ.

    A DC upturn event is confirmed when price rises ≥ θ% from the last trough.
    A DC downturn event is confirmed when price falls ≥ θ% from the last peak.
    The OVERSHOOT is the continuation of the trend after a DC confirmation.

    The paper uses 28 DC indicators including:
        TMV  – total movement value
        OSV  – overshoot value (% beyond the DC confirmation point)
        R    – time-adjusted rate of return
        T    – time spent in current trend

    We implement the core DC signal logic + OSV confirmation,
    plus the full indicator suite for logging/analysis.

Key result from paper:
    DC-based strategies statistically outperform MACD, OBV, and MTM
    across 110 stocks / 10 international markets (Friedman test p < 0.001).
    DC is regime-agnostic — works in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from strategies.strategy_engine import register_strategy
from utils.logger import logger


# ── Core DC Event Detector ────────────────────────────────────────────

def compute_dc_events(prices: np.ndarray, theta: float = 0.01) -> dict:
    """
    Detect all DC events in a price series.

    Args:
        prices: 1-D numpy array of close prices (chronological)
        theta:  threshold for a 'significant' move, e.g. 0.01 = 1%
                Lower θ → more signals, more noise
                Higher θ → fewer signals, higher conviction

    Returns dict with:
        trend          – current trend direction (+1 up, -1 down)
        extreme        – current extreme price (peak or trough)
        last_dc_price  – price at which last DC was confirmed
        last_dc_idx    – bar index of last DC confirmation
        osv            – current Overshoot Value (% beyond DC confirm)
        tmv            – Total Movement Value since last extreme
        dc_events      – list of (idx, direction, price) DC confirmations
        os_events      – list of (idx, direction, price) overshoot bars
        time_in_trend  – bars elapsed since last DC confirmation
        r              – time-adjusted return (TMV / time_in_trend)
    """
    if len(prices) < 2:
        return _empty_dc()

    trend         = 0           # 0 = undefined, +1 = up, -1 = down
    extreme       = prices[0]
    last_dc_price = prices[0]
    last_dc_idx   = 0
    dc_events     = []
    os_events     = []

    for i in range(1, len(prices)):
        p = prices[i]

        if trend >= 0:   # looking for a downward DC
            if p <= extreme * (1.0 - theta):
                # Downward DC confirmed
                trend         = -1
                last_dc_price = p
                last_dc_idx   = i
                extreme       = p
                dc_events.append((i, -1, p))
            else:
                extreme = max(extreme, p)
                os_events.append((i, trend, p))

        if trend <= 0:   # looking for an upward DC
            if p >= extreme * (1.0 + theta):
                # Upward DC confirmed
                trend         = 1
                last_dc_price = p
                last_dc_idx   = i
                extreme       = p
                dc_events.append((i, 1, p))
            else:
                extreme = min(extreme, p) if trend == -1 else extreme
                if trend == -1:
                    os_events.append((i, trend, p))

    # Current indicators
    current        = prices[-1]
    time_in_trend  = len(prices) - 1 - last_dc_idx
    osv            = ((current - last_dc_price) / last_dc_price
                      if last_dc_price != 0 else 0.0)
    tmv            = abs(current - extreme) / extreme if extreme != 0 else 0.0
    r              = osv / max(time_in_trend, 1)

    return {
        "trend":         trend,
        "extreme":       extreme,
        "last_dc_price": last_dc_price,
        "last_dc_idx":   last_dc_idx,
        "osv":           osv,
        "tmv":           tmv,
        "dc_events":     dc_events,
        "os_events":     os_events,
        "time_in_trend": time_in_trend,
        "r":             r,
    }


def _empty_dc() -> dict:
    return {
        "trend": 0, "extreme": 0, "last_dc_price": 0, "last_dc_idx": 0,
        "osv": 0, "tmv": 0, "dc_events": [], "os_events": [],
        "time_in_trend": 0, "r": 0,
    }


# ── DC Indicator Suite (subset from Table 1 of paper) ─────────────────

def add_dc_indicators(df: pd.DataFrame, theta: float = 0.01) -> pd.DataFrame:
    """
    Compute rolling DC indicators and append to dataframe.

    Implements the core DC indicators from Table 1 of Long et al. (2026):
        osv          – current overshoot value
        tmv          – total movement value
        dc_trend     – current DC trend direction (+1 / -1 / 0)
        dc_r         – time-adjusted rate of return
        dc_time      – bars in current trend
        dc_n_10      – number of DC events in last 10 bars
        dc_n_20      – number of DC events in last 20 bars
        dc_n_50      – number of DC events in last 50 bars
    """
    df = df.copy()
    prices = df["close"].values
    n      = len(prices)

    osv_arr   = np.zeros(n)
    tmv_arr   = np.zeros(n)
    trend_arr = np.zeros(n, dtype=int)
    r_arr     = np.zeros(n)
    time_arr  = np.zeros(n, dtype=int)

    for i in range(10, n):
        dc = compute_dc_events(prices[: i + 1], theta)
        osv_arr[i]   = dc["osv"]
        tmv_arr[i]   = dc["tmv"]
        trend_arr[i] = dc["trend"]
        r_arr[i]     = dc["r"]
        time_arr[i]  = dc["time_in_trend"]

    df["dc_osv"]   = osv_arr
    df["dc_tmv"]   = tmv_arr
    df["dc_trend"] = trend_arr
    df["dc_r"]     = r_arr
    df["dc_time"]  = time_arr

    # Rolling DC event counts (n_10, n_20, n_50)
    # Mark bars where a DC event was confirmed by checking trend flips
    flips = (pd.Series(trend_arr).diff().abs() > 0).astype(int)
    df["dc_n_10"] = flips.rolling(10).sum().fillna(0).astype(int)
    df["dc_n_20"] = flips.rolling(20).sum().fillna(0).astype(int)
    df["dc_n_50"] = flips.rolling(50).sum().fillna(0).astype(int)

    return df


# ── Registered Strategy Plugin ────────────────────────────────────────

@register_strategy("DirectionalChange", weight=2.0)
def dc_signal(df: pd.DataFrame, theta: float = 0.01) -> str:
    """
    DC signal with Overshoot Value confirmation.

    Signal logic (from paper Section 4 / GP terminal set):
        BUY  — upward DC confirmed + positive OSV overshoot
               (price momentum continuing past the DC event)
        SELL — downward DC confirmed + negative OSV overshoot
        HOLD — trend undefined or insufficient overshoot

    The overshoot confirmation (OSV > θ × 0.5) filters out
    very fresh DC events that haven't built momentum yet.
    This reduces whipsaw trades in choppy conditions.

    Weight=2.0 because DC outperformed all TA benchmarks in the paper
    (Friedman test, p < 0.001 across all 10 markets).
    """
    if len(df) < 50:
        return "HOLD"

    prices = df["close"].values
    dc     = compute_dc_events(prices, theta=theta)

    trend = dc["trend"]
    osv   = dc["osv"]

    # BUY: uptrend confirmed + meaningful positive overshoot
    if trend == 1 and osv > theta * 0.5:
        return "BUY"

    # SELL: downtrend confirmed + meaningful negative overshoot
    if trend == -1 and osv < -theta * 0.5:
        return "SELL"

    return "HOLD"


# ── generate_signals() shim — for backtester compatibility ────────────

def generate_signals(df: pd.DataFrame, theta: float = 0.01) -> pd.DataFrame:
    """
    Compatibility shim so DC can be used in the backtest pipeline
    (which expects a df with 'signal' and 'crossover' columns).

    Also attaches the full DC indicator suite for analysis.
    """
    df = add_dc_indicators(df, theta=theta)
    prices = df["close"].values
    n      = len(prices)

    signals    = np.zeros(n, dtype=np.int8)
    crossovers = np.zeros(n, dtype=np.int8)
    prev_sig   = 0

    for i in range(50, n):
        dc  = compute_dc_events(prices[:i + 1], theta=theta)
        osv = dc["osv"]
        tr  = dc["trend"]

        if tr == 1 and osv > theta * 0.5:
            sig = 1
        elif tr == -1 and osv < -theta * 0.5:
            sig = -1
        else:
            sig = prev_sig   # maintain current regime

        signals[i] = sig

        # Crossover = first bar the direction changes
        if sig == 1 and prev_sig <= 0 and i > 50:
            crossovers[i] = 1
        elif sig == -1 and prev_sig >= 0 and i > 50:
            crossovers[i] = -1

        prev_sig = sig

    df = df.copy()
    df["signal"]    = signals
    df["crossover"] = crossovers
    return df