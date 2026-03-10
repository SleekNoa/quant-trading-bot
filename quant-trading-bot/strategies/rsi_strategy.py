import pandas as pd
import numpy as np
from config.settings import RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT   # add these to settings.py

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """RSI(14) mean-reversion strategy – very popular for stocks."""
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=RSI_PERIOD).mean()
    loss = -delta.clip(upper=0).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['signal'] = np.where(df['rsi'].isna(), 0,
        np.where(df['rsi'] < RSI_OVERSOLD, 1,
                 np.where(df['rsi'] > RSI_OVERBOUGHT, -1, 0))).astype(np.int8)

    prev_valid = df['rsi'].shift(1).notna()
    df['crossover'] = 0
    df.loc[(df['signal'] == 1) & (df['signal'].shift(1) <= 0) & prev_valid, 'crossover'] = 1
    df.loc[(df['signal'] == -1) & (df['signal'].shift(1) >= 0) & prev_valid, 'crossover'] = -1

    return df