"""
genetic/terminals.py — Terminal Set for GP Trees
==================================================
Defines and normalises the full indicator set used as GP tree leaves.

Source: Long et al. (2026) Tables 1 & 2
    - Table 1: 28 Directional Change (DC) indicators
    - Table 2: 28 Physical-time Technical Analysis (TA) indicators

We implement the subset that integrates with the existing pipeline:
    DC indicators  : dc_osv, dc_tmv, dc_r, dc_time, dc_n_10/20/50 (7)
    TA indicators  : rsi, macd_norm, adx, stoch_k, stoch_d,
                     bb_pct, obv_norm, cci, atr_norm, willr,
                     ema3_norm, ema5_norm, ema10_norm,
                     ma10_norm, ma20_norm, ma30_norm              (16)

All terminals are normalised to [0, 1] using per-dataset min-max scaling
so that ephemeral random constants (ERCs) remain meaningful comparators.

Usage
-----
    from genetic.terminals import build_terminal_array, TERMINAL_NAMES

    norm_matrix = build_terminal_array(df)  # shape (n_bars, n_terminals)
    # norm_matrix[i] is a dict {name: value} for bar i
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict

# ── Terminal name list ─────────────────────────────────────────────────────────

# DC indicators (computed by directional_change_strategy.add_dc_indicators)
DC_TERMINALS: List[str] = [
    "dc_osv",      # Overshoot Value
    "dc_tmv",      # Total Movement Value
    "dc_r",        # Time-adjusted rate of return
    "dc_time",     # Bars in current DC trend
    "dc_n_10",     # DC event count (10 bars)
    "dc_n_20",     # DC event count (20 bars)
    "dc_n_50",     # DC event count (50 bars)
]

# Technical Analysis indicators
TA_TERMINALS: List[str] = [
    "rsi",          # RSI(14) — already in df [0, 100] → will normalise
    "macd_norm",    # MACD line, normalised
    "adx",          # ADX(14) — trend strength [0, 100]
    "stoch_k",      # Stochastic %K [0, 100]
    "stoch_d",      # Stochastic %D [0, 100]
    "bb_pct",       # Price position within Bollinger Bands [0, 1]
    "obv_norm",     # OBV normalised
    "cci",          # Commodity Channel Index (computed here)
    "atr_norm",     # ATR normalised
    "willr",        # Williams %R, normalised to [0, 1]
    "ema3_norm",    # EMA(3) / close — near 1.0 when trending
    "ema5_norm",    # EMA(5) / close
    "ema10_norm",   # EMA(10) / close
    "ma10_norm",    # MA(10) / close
    "ma20_norm",    # MA(20) / close
    "ma30_norm",    # MA(30) / close
]

TERMINAL_NAMES: List[str] = DC_TERMINALS + TA_TERMINALS


# ── Compute supplemental indicators not in main pipeline ──────────────────────

def _compute_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma) / (0.015 * md.replace(0, np.nan))
    return cci.fillna(0)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(0)


def _compute_willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R, mapped to [0, 1]  (0=oversold, 1=overbought)."""
    highest_high = df["high"].rolling(period).max()
    lowest_low   = df["low"].rolling(period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    willr = (highest_high - df["close"]) / denom   # raw: 0 = overbought, 1 = oversold
    return (1 - willr.fillna(0.5))                 # flip so 1 = overbought


def _minmax(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Scale a series to [0, 1] using global min-max."""
    lo, hi = series.min(), series.max()
    if abs(hi - lo) < eps:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - lo) / (hi - lo)


# ── Main function: build normalised terminal matrix ────────────────────────────

def build_terminal_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a (n_bars × n_terminals) float32 matrix of normalised [0,1] values.

    Parameters
    ----------
    df : DataFrame with OHLCV + all indicator columns from main pipeline
         Required columns: open, high, low, close, volume,
                           rsi, macd, adx, stoch_k, stoch_d,
                           bb_upper, bb_lower, obv,
                           dc_osv, dc_tmv, dc_r, dc_time,
                           dc_n_10, dc_n_20, dc_n_50

    Returns
    -------
    matrix : np.ndarray  shape (n_bars, len(TERMINAL_NAMES))
    norms  : dict  {name: (lo, hi)} for inverse-transform if needed
    """
    n = len(df)
    matrix = np.zeros((n, len(TERMINAL_NAMES)), dtype=np.float32)
    norms: Dict[str, tuple] = {}

    def _col(series: pd.Series, idx: int) -> None:
        s = series.ffill().fillna(0)
        lo, hi = float(s.min()), float(s.max())
        norms[TERMINAL_NAMES[idx]] = (lo, hi)
        denom = hi - lo if abs(hi - lo) > 1e-9 else 1.0
        matrix[:, idx] = ((s - lo) / denom).values

    for i, name in enumerate(DC_TERMINALS):
        if name in df.columns:
            _col(df[name], i)
        # else: leave as 0.5 (no DC indicators computed yet)

    close = df["close"]

    # RSI
    _col(df["rsi"] if "rsi" in df.columns else pd.Series(50, index=df.index),
         TERMINAL_NAMES.index("rsi"))

    # MACD
    _col(df["macd"] if "macd" in df.columns else pd.Series(0, index=df.index),
         TERMINAL_NAMES.index("macd_norm"))

    # ADX
    _col(df["adx"] if "adx" in df.columns else pd.Series(25, index=df.index),
         TERMINAL_NAMES.index("adx"))

    # Stochastic
    _col(df["stoch_k"] if "stoch_k" in df.columns else pd.Series(50, index=df.index),
         TERMINAL_NAMES.index("stoch_k"))
    _col(df["stoch_d"] if "stoch_d" in df.columns else pd.Series(50, index=df.index),
         TERMINAL_NAMES.index("stoch_d"))

    # Bollinger %B — where is price within the band?
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        denom = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan).fillna(1)
        bb_pct = ((close - df["bb_lower"]) / denom).clip(0, 1)
    else:
        bb_pct = pd.Series(0.5, index=df.index)
    _col(bb_pct, TERMINAL_NAMES.index("bb_pct"))

    # OBV
    _col(df["obv"] if "obv" in df.columns else pd.Series(0, index=df.index),
         TERMINAL_NAMES.index("obv_norm"))

    # CCI
    _col(_compute_cci(df), TERMINAL_NAMES.index("cci"))

    # ATR
    _col(_compute_atr(df), TERMINAL_NAMES.index("atr_norm"))

    # Williams %R
    _col(_compute_willr(df), TERMINAL_NAMES.index("willr"))

    # EMA ratios  (price / ema — centred near 1; normalised)
    for span, name in [(3, "ema3_norm"), (5, "ema5_norm"), (10, "ema10_norm")]:
        ema = close.ewm(span=span, adjust=False).mean()
        ratio = (close / ema.replace(0, np.nan)).fillna(1)
        _col(ratio, TERMINAL_NAMES.index(name))

    # MA ratios
    for window, name in [(10, "ma10_norm"), (20, "ma20_norm"), (30, "ma30_norm")]:
        ma = close.rolling(window).mean().fillna(close)
        ratio = (close / ma.replace(0, np.nan)).fillna(1)
        _col(ratio, TERMINAL_NAMES.index(name))

    return matrix, norms


def row_to_dict(matrix: np.ndarray, row_idx: int) -> Dict[str, float]:
    """Convert a single row of the terminal matrix to a {name: value} dict."""
    return {name: float(matrix[row_idx, i]) for i, name in enumerate(TERMINAL_NAMES)}