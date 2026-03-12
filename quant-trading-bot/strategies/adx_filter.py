"""
ADX Trend Filter
================
The Average Directional Index (ADX) measures trend STRENGTH, not direction.
It answers: "Is the market trending at all right now?"

ADX < 25  →  choppy / sideways — momentum signals are unreliable
ADX ≥ 25  →  trending — momentum signals (MACD, RSI, etc.) are valid

Source concept: Liberated Stock Trader categorises ADX as a "trend strength"
indicator, explicitly separate from directional indicators. The Cheat Sheet
places it in the "trend" column, making it ideal as a filter gate rather than
a standalone signal.

Why this matters for your bot:
    Without ADX, MACD fires during sideways markets and produces whipsaw trades.
    A MACD crossover in a ranging market is a false signal ~60% of the time.
    Gating on ADX > 25 keeps you out of those conditions entirely.
"""

import numpy as np
import pandas as pd
from config.settings import ADX_PERIOD, ADX_THRESHOLD


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ADX, +DI, -DI columns to the dataframe.

    Requires: high, low, close columns.
    Adds:
        adx          – trend strength (0–100, >25 = trending)
        adx_trending – bool: True when ADX >= ADX_THRESHOLD
    """
    required = ["high", "low", "close"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"ADX filter requires columns: {required}")

    df = df.copy()
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    dm_plus  = np.where((high - high.shift(1)) > (low.shift(1) - low),
                        np.maximum(high - high.shift(1), 0), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                        np.maximum(low.shift(1) - low, 0), 0)

    dm_plus_s  = pd.Series(dm_plus,  index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()
    dm_minus_s = pd.Series(dm_minus, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()
    atr        = tr.ewm(span=ADX_PERIOD, adjust=False).mean()

    di_plus  = 100 * dm_plus_s  / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus_s / atr.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)

    df["adx"]          = dx.ewm(span=ADX_PERIOD, adjust=False).mean().round(2)
    df["adx_trending"] = df["adx"] >= ADX_THRESHOLD

    return df


def apply_adx_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gates crossover signals: suppresses any crossover where ADX < threshold.
    The signal column (trend regime) is preserved — only the crossover EVENT
    is suppressed, so the bot won't enter a trade in a ranging market.

    A suppressed crossover is set to 0 (no action this bar).
    """
    if "adx_trending" not in df.columns:
        df = add_adx(df)

    df = df.copy()
    # Only allow crossovers when the market is actually trending
    df.loc[~df["adx_trending"], "crossover"] = 0
    return df