import pandas as pd
import numpy as np
from config.settings import BB_PERIOD, BB_STD   # add to settings.py

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Bands (20,2) – buy on lower band, sell on upper band (mean reversion)."""
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    df = df.copy()
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_STD * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - BB_STD * df['bb_std']

    df['signal'] = np.where(df['close'].isna() | df['bb_lower'].isna(), 0,
        np.where(df['close'] < df['bb_lower'], 1,
                 np.where(df['close'] > df['bb_upper'], -1, 0))).astype(np.int8)

    prev_valid = df['bb_lower'].shift(1).notna() & df['bb_upper'].shift(1).notna()
    df['crossover'] = 0
    df.loc[(df['signal'] == 1) & (df['signal'].shift(1) <= 0) & prev_valid, 'crossover'] = 1
    df.loc[(df['signal'] == -1) & (df['signal'].shift(1) >= 0) & prev_valid, 'crossover'] = -1

    return df