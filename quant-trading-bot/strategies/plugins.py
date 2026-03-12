"""
strategies/plugins.py — Strategy Plugin Registrations
======================================================
Wraps all existing generate_signals() strategies as engine plugins.

Each function:
    1. Reads the latest bar from the pre-computed indicator columns
       (added by the generate_signals() call in main.py)
    2. Returns "BUY" | "SELL" | "HOLD"
    3. Self-registers via @register_strategy so main.py never needs
       to import these individually

Weights reflect the relative evidence strength from:
    Long et al. (2026) Table 11 — MACD and OBV-based strategies
    underperformed DC by a statistically significant margin.
    Mean-reversion strategies (RSI, Bollinger) add independent signal
    in ranging regimes (see regime multipliers in strategy_engine.py).
"""

import pandas as pd
import numpy as np
from strategies.strategy_engine import register_strategy


# ── MACD ──────────────────────────────────────────────────────────────

@register_strategy("MACD", weight=1.0)
def macd_plugin(df: pd.DataFrame) -> str:
    """
    MACD crossover signal from pre-computed columns.
    Uses current bar's signal column (1=bullish, -1=bearish).
    """
    if "macd" not in df.columns or "macd_signal" not in df.columns:
        return "HOLD"
    latest = df.iloc[-1]
    if pd.isna(latest["macd"]) or pd.isna(latest["macd_signal"]):
        return "HOLD"
    if latest["macd"] > latest["macd_signal"]:
        return "BUY"
    if latest["macd"] < latest["macd_signal"]:
        return "SELL"
    return "HOLD"


# ── RSI ───────────────────────────────────────────────────────────────

@register_strategy("RSI", weight=1.0)
def rsi_plugin(df: pd.DataFrame) -> str:
    """
    RSI mean-reversion signal.
    Oversold (<30) → BUY,  Overbought (>70) → SELL.
    """
    if "rsi" not in df.columns:
        return "HOLD"
    latest = df.iloc[-1]
    if pd.isna(latest["rsi"]):
        return "HOLD"
    rsi = float(latest["rsi"])
    if rsi < 30:
        return "BUY"
    if rsi > 70:
        return "SELL"
    return "HOLD"


# ── Bollinger Bands ───────────────────────────────────────────────────

@register_strategy("Bollinger", weight=1.0)
def bollinger_plugin(df: pd.DataFrame) -> str:
    """
    Bollinger mean-reversion: touch lower band = BUY, upper = SELL.
    """
    if "bb_lower" not in df.columns or "bb_upper" not in df.columns:
        return "HOLD"
    latest = df.iloc[-1]
    if pd.isna(latest.get("bb_lower")) or pd.isna(latest.get("bb_upper")):
        return "HOLD"
    close = float(latest["close"])
    if close < float(latest["bb_lower"]):
        return "BUY"
    if close > float(latest["bb_upper"]):
        return "SELL"
    return "HOLD"


# ── Stochastic ────────────────────────────────────────────────────────

@register_strategy("Stochastic", weight=1.0)
def stochastic_plugin(df: pd.DataFrame) -> str:
    """
    Stochastic %K/%D crossover signal.
    %K crosses above %D → BUY, crosses below → SELL.
    """
    if "%K" not in df.columns or "%D" not in df.columns:
        return "HOLD"
    latest = df.iloc[-1]
    if pd.isna(latest.get("%K")) or pd.isna(latest.get("%D")):
        return "HOLD"
    k = float(latest["%K"])
    d = float(latest["%D"])
    if k > d:
        return "BUY"
    if k < d:
        return "SELL"
    return "HOLD"


# ── SMA (Moving Average Crossover) ────────────────────────────────────

@register_strategy("SMA", weight=0.8)
def sma_plugin(df: pd.DataFrame) -> str:
    """
    Simple MA crossover: short_ma > long_ma → BUY, else SELL.
    Lower weight because SMA is the laggiest of the group.
    """
    if "short_ma" not in df.columns or "long_ma" not in df.columns:
        return "HOLD"
    latest = df.iloc[-1]
    if pd.isna(latest.get("short_ma")) or pd.isna(latest.get("long_ma")):
        return "HOLD"
    if float(latest["short_ma"]) > float(latest["long_ma"]):
        return "BUY"
    if float(latest["short_ma"]) < float(latest["long_ma"]):
        return "SELL"
    return "HOLD"


# ── OBV Trend ─────────────────────────────────────────────────────────

@register_strategy("OBV", weight=0.8)
def obv_plugin(df: pd.DataFrame) -> str:
    """
    OBV trend confirmation: OBV above its MA = buying pressure = BUY.
    Long et al. use OBV as one of the 28 TA benchmark indicators.
    """
    if "obv_rising" not in df.columns:
        return "HOLD"
    latest = df.iloc[-1]
    rising = latest.get("obv_rising", False)
    if rising:
        return "BUY"
    return "SELL"