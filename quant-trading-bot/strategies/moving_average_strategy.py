import pandas as pd
import numpy as np
from config.settings import SHORT_WINDOW, LONG_WINDOW


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate moving average crossover signals using standardized OHLCV DataFrame.

    Expected input columns:
        - 'close'   (adjusted or unadjusted closing price)

    Output columns added:
        - 'short_ma'    : simple moving average (SHORT_WINDOW)
        - 'long_ma'     : simple moving average (LONG_WINDOW)
        - 'signal'      :  1 = bullish (short > long), -1 = bearish (short < long), 0 = neutral/undefined
        - 'crossover'   : +1 = fresh bullish cross, -1 = fresh bearish cross, 0 = no cross this bar

    Rules:
    - Signals are only generated when **both** MAs are non-NaN
    - Crossover is only flagged when the inequality direction reverses
      (from bearish/neutral → bullish, or bullish/neutral → bearish)
    - Vectorized, NaN-safe implementation

    Raises:
        ValueError: if required 'close' column is missing
    """
    if 'close' not in df.columns:
        raise ValueError(
            "DataFrame must contain 'close' column. "
            "Use standardized OHLCV format (see market_data.py)."
        )

    df = df.copy()  # prevent SettingWithCopyWarning

    # ── Calculate moving averages ────────────────────────────────────────
    df['short_ma'] = df['close'].rolling(window=SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    df['long_ma']  = df['close'].rolling(window=LONG_WINDOW, min_periods=LONG_WINDOW).mean()

    # ── Create current regime (state) ─────────────────────────────────────
    #  1 = short > long    (bullish)
    # -1 = short < long    (bearish)
    #  0 = one or both MAs are NaN, or equal (very rare)
    df['signal'] = np.where(
        df['short_ma'].isna() | df['long_ma'].isna(), 0,
        np.where(df['short_ma'] > df['long_ma'], 1,
                 np.where(df['short_ma'] < df['long_ma'], -1, 0))
    ).astype(np.int8)

    # ── Detect crossovers (events) ────────────────────────────────────────
    # Only when both MAs were valid on previous bar too
    prev_short_valid = df['short_ma'].shift(1).notna()
    prev_long_valid  = df['long_ma'].shift(1).notna()
    prev_valid = prev_short_valid & prev_long_valid

    df['crossover'] = 0

    # Bullish cross: was ≤ before, now >  (and both were valid before)
    bullish_cross = (
        (df['signal'] == 1) &
        (df['signal'].shift(1) <= 0) &
        prev_valid
    )
    df.loc[bullish_cross, 'crossover'] = 1

    # Bearish cross: was ≥ before, now <  (and both were valid before)
    bearish_cross = (
        (df['signal'] == -1) &
        (df['signal'].shift(1) >= 0) &
        prev_valid
    )
    df.loc[bearish_cross, 'crossover'] = -1

    # Optional: drop temporary columns if you want cleaner output
    # df.drop(columns=['short_valid', 'long_valid'], errors='ignore', inplace=True)

    return df