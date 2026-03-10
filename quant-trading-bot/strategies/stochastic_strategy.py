import pandas as pd
import numpy as np
from config.settings import STOCH_K, STOCH_D   # add to settings.py

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic (14,3) – very responsive momentum strategy."""
    required = ['high', 'low', 'close']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame must contain {required} columns.")

    df = df.copy()
    lowest_low = df['low'].rolling(STOCH_K).min()
    highest_high = df['high'].rolling(STOCH_K).max()
    df['%K'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    df['%D'] = df['%K'].rolling(STOCH_D).mean()

    df['signal'] = np.where(df['%K'].isna() | df['%D'].isna(), 0,
        np.where(df['%K'] > df['%D'], 1, -1)).astype(np.int8)

    prev_valid = df['%K'].shift(1).notna() & df['%D'].shift(1).notna()
    df['crossover'] = 0
    df.loc[(df['signal'] == 1) & (df['signal'].shift(1) <= 0) & prev_valid, 'crossover'] = 1
    df.loc[(df['signal'] == -1) & (df['signal'].shift(1) >= 0) & prev_valid, 'crossover'] = -1

    return df