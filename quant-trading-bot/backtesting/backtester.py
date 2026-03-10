import pandas as pd
from config.settings import INITIAL_CAPITAL
from risk.risk_manager import full_position_size


def backtest(df: pd.DataFrame) -> dict:
    """
    Simulate the MA crossover strategy over historical data.

    Uses crossover column for entry/exit timing so we only act on
    the bar where the signal *changes* (not every bar it stays the same).

    Returns a dict with:
        final_value  – closing portfolio value ($)
        return_pct   – percentage return vs INITIAL_CAPITAL
        trades       – list of trade dicts
        win_rate     – % of closed trades that were profitable
        buy_hold     – what a simple buy-and-hold would have returned
        n_trades     – total number of trades (buy + sell)
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column (standardized OHLCV format)")
    if "crossover" not in df.columns:
        raise ValueError("DataFrame must contain 'crossover' column from generate_signals()")

    cash        = float(INITIAL_CAPITAL)
    position    = 0             # shares held
    entry_price = 0.0
    trades      = []

    for i in range(len(df)):
        price     = df["close"].iloc[i]
        crossover = df["crossover"].iloc[i]
        date      = df.index[i].strftime("%Y-%m-%d")

        # BUY crossover — go long (only if we have cash and no position)
        if crossover == 1 and cash > 0 and position == 0:
            shares = full_position_size(cash, price)
            if shares > 0:
                cost = shares * price
                if cost <= cash:  # safety check
                    cash       -= cost
                    position    = shares
                    entry_price = price
                    trades.append({
                        "date": date,
                        "type": "BUY",
                        "price": price,
                        "shares": shares,
                        "value": cost,
                        "pnl": None,
                    })

        # SELL crossover — close long
        elif crossover == -1 and position > 0:
            proceeds = position * price
            pnl = (price - entry_price) * position
            cash += proceeds
            trades.append({
                "date": date,
                "type": "SELL",
                "price": price,
                "shares": position,
                "value": proceeds,
                "pnl": pnl,
            })
            position    = 0
            entry_price = 0.0

    # Final portfolio value
    final_value = cash + (position * df["close"].iloc[-1] if len(df) > 0 else 0)

    return_pct = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                  if INITIAL_CAPITAL > 0 else 0.0)

    # Buy-and-hold benchmark
    if len(df) > 1:
        first_price = df["close"].iloc[0]
        last_price  = df["close"].iloc[-1]
        buy_hold_pct = ((last_price - first_price) / first_price * 100)
    else:
        buy_hold_pct = 0.0

    # Win rate (only on closed trades)
    closed_trades = [t for t in trades if t["type"] == "SELL"]
    wins          = [t for t in closed_trades if t["pnl"] is not None and t["pnl"] > 0]
    win_rate      = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0

    return {
        "final_value": round(final_value, 2),
        "return_pct":  round(return_pct, 2),
        "buy_hold":    round(buy_hold_pct, 2),
        "win_rate":    round(win_rate, 1),
        "n_trades":    len(trades),
        "trades":      trades,
    }