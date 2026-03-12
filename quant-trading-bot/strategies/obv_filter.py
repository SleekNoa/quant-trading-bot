"""
OBV Volume Confirmation Filter
================================
On Balance Volume (OBV) tracks cumulative buying/selling pressure by adding
volume on up-days and subtracting it on down-days.

Why this matters for your bot:
    Price can be pushed up by a small number of participants short-term.
    Volume cannot be faked the same way — it reflects real money flow.
    A MACD bullish crossover WITH rising OBV means genuine buying pressure
    is backing the move. Without it, the crossover is more likely a fake-out.

    Liberated Stock Trader specifically notes that volume indicators like OBV
    confirm whether price moves are supported by real participation.
    The Cheat Sheet classifies OBV under "volume" — a separate category from
    momentum — making it a true independent confirmation layer.

Implementation:
    obv_rising = OBV is above its own short moving average (5-bar default)
    A BUY crossover is only kept if obv_rising = True
    A SELL crossover is only kept if obv_rising = False (selling pressure)
"""

import pandas as pd
from config.settings import OBV_MA_PERIOD


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds OBV and obv_ma (smoothed OBV) to the dataframe.

    Requires: close, volume columns.
    Adds:
        obv        – raw cumulative OBV
        obv_ma     – OBV moving average (OBV_MA_PERIOD bars)
        obv_rising – True when OBV > obv_ma (buying pressure dominant)
    """
    required = ["close", "volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"OBV filter requires columns: {required}")

    df = df.copy()

    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["obv"]        = (direction * df["volume"]).cumsum()
    df["obv_ma"]     = df["obv"].rolling(OBV_MA_PERIOD).mean()
    df["obv_rising"] = df["obv"] > df["obv_ma"]

    return df


def apply_obv_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confirms crossovers with volume:
        BUY  crossover kept only when OBV is rising  (buyers in control)
        SELL crossover kept only when OBV is falling (sellers in control)

    Any crossover not confirmed by volume is suppressed (set to 0).
    This reduces noise trades significantly, especially in low-volume ranges.
    """
    if "obv_rising" not in df.columns:
        df = add_obv(df)

    df = df.copy()

    # Suppress unconfirmed buys (no volume backing)
    df.loc[(df["crossover"] == 1)  & (~df["obv_rising"]), "crossover"] = 0
    # Suppress unconfirmed sells (no selling pressure)
    df.loc[(df["crossover"] == -1) & (df["obv_rising"]),  "crossover"] = 0

    return df