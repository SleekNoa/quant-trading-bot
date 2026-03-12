import numpy as np
import pandas as pd
from config.settings import INITIAL_CAPITAL
from risk.risk_manager import full_position_size
"""
backtesting/strategy_backtest.py — Per-Strategy Performance Analysis
=====================================================================
Runs a separate backtest for each registered strategy plugin and
returns a comparison table showing which strategies perform best
on the current asset/period.

Inspired by the benchmarking methodology in:
    Long et al. (2026) Table 11 — comparison of MOO3 vs individual
    TA indicators across 110 stocks. We do the same across your
    strategy set so you can see exactly which plugins are adding alpha.

Output metrics per strategy:
    return_pct    – total return vs initial capital
    buy_hold      – passive benchmark
    alpha         – return_pct - buy_hold
    sharpe        – annualised Sharpe ratio
    max_drawdown  – worst peak-to-trough %
    win_rate      – % of closed trades profitable
    n_trades      – number of round-trip trades
    avg_trade_pct – average % gain per trade
"""

import numpy as np
import pandas as pd
from utils.logger import logger
from config.settings import INITIAL_CAPITAL
from risk.risk_manager import full_position_size


def _run_single_backtest(df_with_signals: pd.DataFrame) -> dict:
    """Run a single backtest on a df that already has signal/crossover columns."""
    if "crossover" not in df_with_signals.columns:
        return None

    cash        = float(INITIAL_CAPITAL)
    position    = 0
    entry_price = 0.0
    trades      = []
    daily_values = []
    df          = df_with_signals

    for i in range(len(df)):
        price     = float(df["close"].iloc[i])
        crossover = int(df["crossover"].iloc[i])

        if crossover == 1 and cash > 0 and position == 0:
            shares = full_position_size(cash, price)
            if shares > 0:
                cost         = shares * price
                cash        -= cost
                position     = shares
                entry_price  = price
                trades.append({"type": "BUY", "price": price, "shares": shares})

        elif crossover == -1 and position > 0:
            proceeds    = position * price
            pnl         = (price - entry_price) * position
            cash       += proceeds
            trades.append({"type": "SELL", "price": price, "shares": position, "pnl": pnl})
            position    = 0
            entry_price = 0.0

        daily_values.append(cash + position * price)

    final_value = cash + (position * float(df["close"].iloc[-1]) if len(df) > 0 else 0)
    return_pct  = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    buy_hold    = (float(df["close"].iloc[-1]) - float(df["close"].iloc[0])) / float(df["close"].iloc[0]) * 100

    closed   = [t for t in trades if t["type"] == "SELL"]
    wins     = [t for t in closed if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    trade_pcts    = []
    last_buy_price = None
    for t in trades:
        if t["type"] == "BUY":
            last_buy_price = t["price"]
        elif t["type"] == "SELL" and last_buy_price:
            trade_pcts.append((t["price"] - last_buy_price) / last_buy_price * 100)
            last_buy_price = None
    avg_trade_pct = round(float(np.mean(trade_pcts)), 2) if trade_pcts else 0.0

    vals = np.array(daily_values, dtype=float)
    daily_returns = np.diff(vals) / vals[:-1] if len(vals) > 1 else np.array([0.0])
    sharpe = round(float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)), 2) \
             if daily_returns.std() > 0 else 0.0

    peak = vals[0] if len(vals) > 0 else float(INITIAL_CAPITAL)
    max_dd = 0.0
    for v in vals:
        if v > peak: peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd: max_dd = dd

    return {
        "return_pct":    round(return_pct, 2),
        "buy_hold":      round(buy_hold, 2),
        "alpha":         round(return_pct - buy_hold, 2),
        "sharpe":        sharpe,
        "max_drawdown":  round(max_dd, 2),
        "win_rate":      round(win_rate, 1),
        "n_trades":      len([t for t in trades if t["type"] == "BUY"]),
        "avg_trade_pct": avg_trade_pct,
        "final_value":   round(final_value, 2),
    }


def backtest_all_strategies(df_raw: pd.DataFrame) -> dict:
    """
    Run a backtest for each registered strategy and return a comparison dict.

    Args:
        df_raw: clean OHLCV DataFrame (no indicators pre-computed)

    Returns:
        {strategy_name: metrics_dict, ...}
        Plus a "COMBINED" entry for the multi-strategy consensus.
    """
    from strategies.macd_strategy       import generate_signals as macd_gen
    from strategies.rsi_strategy        import generate_signals as rsi_gen
    from strategies.bollinger_strategy  import generate_signals as bollinger_gen
    from strategies.stochastic_strategy import generate_signals as stochastic_gen
    from strategies.moving_average_strategy import generate_signals as sma_gen
    from strategies.directional_change_strategy import generate_signals as dc_gen

    strategy_generators = {
        "MACD":             macd_gen,
        "RSI":              rsi_gen,
        "Bollinger":        bollinger_gen,
        "Stochastic":       stochastic_gen,
        "SMA":              sma_gen,
        "DirectionalChange": dc_gen,
    }

    results = {}
    for name, gen_fn in strategy_generators.items():
        try:
            df_signals = gen_fn(df_raw.copy())
            metrics    = _run_single_backtest(df_signals)
            if metrics:
                results[name] = metrics
        except Exception as e:
            logger.warning(f"[backtest] Strategy '{name}' failed: {e}")
            results[name] = None

    return results


def print_strategy_comparison(results: dict):
    """Pretty-print a side-by-side strategy performance table."""
    SEP  = "═" * 90
    SEP2 = "─" * 90

    logger.info(SEP)
    logger.info("  PER-STRATEGY PERFORMANCE COMPARISON")
    logger.info(SEP)
    logger.info(
        f"  {'Strategy':<22} {'Return':>8} {'Alpha':>8} {'Sharpe':>7} "
        f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'AvgTrade':>9}"
    )
    logger.info(SEP2)

    # Sort by Sharpe descending
    valid = {k: v for k, v in results.items() if v is not None}
    sorted_results = sorted(valid.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    for name, m in sorted_results:
        ret_str  = f"+{m['return_pct']:.1f}%" if m['return_pct'] >= 0 else f"{m['return_pct']:.1f}%"
        alp_str  = f"+{m['alpha']:.1f}%"      if m['alpha'] >= 0      else f"{m['alpha']:.1f}%"
        dd_str   = f"{m['max_drawdown']:.1f}%"

        # Medal for top performers
        rank_icon = ""
        if name == sorted_results[0][0]:  rank_icon = "🥇"
        elif name == sorted_results[1][0] if len(sorted_results) > 1 else False: rank_icon = "🥈"

        logger.info(
            f"  {name + ' ' + rank_icon:<24} {ret_str:>8} {alp_str:>8} "
            f"{m['sharpe']:>7.2f} {dd_str:>8} "
            f"{m['win_rate']:>7.1f}% {m['n_trades']:>7}  {m['avg_trade_pct']:>8.2f}%"
        )

    buy_hold = list(valid.values())[0]["buy_hold"] if valid else 0
    logger.info(SEP2)
    logger.info(f"  {'Buy & Hold (passive)':<22} {buy_hold:>+8.1f}%  (benchmark)")
    logger.info(SEP)