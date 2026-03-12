import numpy as np
import pandas as pd
from config.settings import INITIAL_CAPITAL
from risk.risk_manager import full_position_size


def backtest(df: pd.DataFrame) -> dict:
    """
    Simulate strategy over historical data.

    Returns:
        final_value   – closing portfolio value ($)
        return_pct    – % return vs INITIAL_CAPITAL
        trades        – list of trade dicts
        win_rate      – % of closed trades that were profitable
        buy_hold      – simple buy-and-hold return (%)
        n_trades      – total signals fired (buy + sell)
        sharpe        – annualised Sharpe ratio (daily returns, rf=0)
        max_drawdown  – worst peak-to-trough decline (%)
        avg_trade_pct – average % gain/loss per closed trade
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    if "crossover" not in df.columns:
        raise ValueError("DataFrame must contain 'crossover' column from generate_signals()")

    cash        = float(INITIAL_CAPITAL)
    position    = 0
    entry_price = 0.0
    trades      = []
    daily_values = []

    for i in range(len(df)):
        price     = df["close"].iloc[i]
        crossover = df["crossover"].iloc[i]
        date      = df.index[i].strftime("%Y-%m-%d")

        if crossover == 1 and cash > 0 and position == 0:
            shares = full_position_size(cash, price)
            if shares > 0:
                cost = shares * price
                if cost <= cash:
                    cash        -= cost
                    position     = shares
                    entry_price  = price
                    trades.append({
                        "date":   date,
                        "type":   "BUY",
                        "price":  price,
                        "shares": shares,
                        "value":  cost,
                        "pnl":    None,
                    })

        elif crossover == -1 and position > 0:
            proceeds = position * price
            pnl      = (price - entry_price) * position
            cash    += proceeds
            trades.append({
                "date":   date,
                "type":   "SELL",
                "price":  price,
                "shares": position,
                "value":  proceeds,
                "pnl":    pnl,
            })
            position    = 0
            entry_price = 0.0

        daily_values.append(cash + position * price)

    final_value = cash + (position * df["close"].iloc[-1] if len(df) > 0 else 0)
    return_pct  = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 if INITIAL_CAPITAL > 0 else 0.0

    if len(df) > 1:
        buy_hold_pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    else:
        buy_hold_pct = 0.0

    closed   = [t for t in trades if t["type"] == "SELL"]
    wins     = [t for t in closed if t["pnl"] is not None and t["pnl"] > 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    # Average % per closed trade
    buy_prices = {}
    trade_pcts = []
    for t in trades:
        if t["type"] == "BUY":
            buy_prices[t["date"]] = t["price"]
    last_buy_price = None
    for t in trades:
        if t["type"] == "BUY":
            last_buy_price = t["price"]
        elif t["type"] == "SELL" and last_buy_price:
            trade_pcts.append((t["price"] - last_buy_price) / last_buy_price * 100)
            last_buy_price = None
    avg_trade_pct = round(float(np.mean(trade_pcts)), 2) if trade_pcts else 0.0

    # Sharpe ratio (annualised, risk-free = 0)
    vals = np.array(daily_values, dtype=float)
    daily_returns = np.diff(vals) / vals[:-1] if len(vals) > 1 else np.array([0.0])
    if daily_returns.std() > 0:
        sharpe = round(float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)), 2)
    else:
        sharpe = 0.0

    # Max drawdown
    peak   = vals[0] if len(vals) > 0 else float(INITIAL_CAPITAL)
    max_dd = 0.0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    return {
        "final_value":   round(final_value,   2),
        "return_pct":    round(return_pct,     2),
        "buy_hold":      round(buy_hold_pct,   2),
        "win_rate":      round(win_rate,       1),
        "n_trades":      len(trades),
        "trades":        trades,
        "sharpe":        sharpe,
        "max_drawdown":  round(max_dd,         2),
        "avg_trade_pct": avg_trade_pct,
    }