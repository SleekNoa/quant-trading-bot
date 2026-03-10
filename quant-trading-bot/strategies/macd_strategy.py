import pandas as pd
import numpy as np
from config.settings import MACD_FAST, MACD_SLOW, MACD_SIGNAL   # add to settings.py

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """MACD(12,26,9) – one of the most widely used institutional signals."""
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    df = df.copy()
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()

    df['signal'] = np.where(df['macd'].isna() | df['macd_signal'].isna(), 0,
        np.where(df['macd'] > df['macd_signal'], 1, -1)).astype(np.int8)

    prev_valid = (df['macd'].shift(1).notna()) & (df['macd_signal'].shift(1).notna())
    df['crossover'] = 0
    df.loc[(df['signal'] == 1) & (df['signal'].shift(1) <= 0) & prev_valid, 'crossover'] = 1
    df.loc[(df['signal'] == -1) & (df['signal'].shift(1) >= 0) & prev_valid, 'crossover'] = -1

    return df