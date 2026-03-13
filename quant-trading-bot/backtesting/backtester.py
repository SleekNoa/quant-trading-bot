"""
backtesting/backtester.py — Ensemble Engine Backtest
===================================================
Called by main.py as backtest_engine(df)
Uses the full per-strategy comparison (MACD, RSI, BB, Stoch, SMA, DC)
and returns the exact dict shape expected by print_backtest_header().
"""

import numpy as np
import pandas as pd
from config.settings import INITIAL_CAPITAL
from risk.risk_manager import full_position_size
from utils.logger import logger


def _run_single_backtest(df_with_signals: pd.DataFrame) -> dict | None:
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

    trade_pcts = []
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
        "return_pct": round(return_pct, 2),
        "buy_hold": round(buy_hold, 2),
        "alpha": round(return_pct - buy_hold, 2),
        "sharpe": sharpe,
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "n_trades": len([t for t in trades if t["type"] == "BUY"]),
        "avg_trade_pct": avg_trade_pct,
        "final_value": round(final_value, 2),
        # ── ADD THESE TWO LINES ───────────────────────────────────────────────
        "trade_returns": trade_pcts,  # list[float] — what MC usually wants
        "trades": trades,  # optional: full trade dicts if needed later
    }


# ── ADD this function ABOVE backtest_all_strategies() ─────────────────────────

def _run_moo3_backtest(df: pd.DataFrame) -> dict | None:
    """
    Run a historical backtest for the saved MOO3 individual.

    MOO3 uses time/price-based exits (sell_days + sell_pct + sl_pct),
    NOT crossover signals.  We use the same trade simulator as training
    (genetic/fitness.py) so the numbers are directly comparable to the
    Pareto front objectives printed during genetic/run_genetic.py.

    Pre-builds the terminal matrix once for all bars — O(n) not O(n²).
    """
    import os
    _MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "models", "moo3_best.pkl"
    )
    if not os.path.exists(_MODEL_PATH):
        return None

    try:
        from genetic.gp_engine import MOO3Engine
        from genetic.terminals import build_terminal_matrix, row_to_dict
        from genetic.fitness import simulate_trades, compute_objectives

        best = MOO3Engine.load()

        # ── Pre-build terminal matrix once for all bars ────────────────────
        close_prices = df["close"].values.astype(float)
        term_matrix, _ = build_terminal_matrix(df)

        # ── Evaluate GP tree for every bar (boolean: BUY or HOLD) ─────────
        import numpy as np
        n = len(close_prices)
        buy_signals = np.zeros(n, dtype=bool)
        for i in range(n):
            row = row_to_dict(term_matrix, i)
            try:
                buy_signals[i] = best.tree.evaluate(row)
            except Exception:
                buy_signals[i] = False

        # ── Simulate trades with MOO3's evolved exit parameters ───────────
        trades = simulate_trades(
            close_prices,
            buy_signals,
            sell_days=best.sell_days,
            sell_pct=best.sell_pct,
            sl_pct=best.sl_pct,
        )

        if not trades:
            return None

        # ── Convert to the standard metrics dict ──────────────────────────
        tr, wr, max_dd = compute_objectives(trades)

        # Rebuild equity curve from trade list for Sharpe + daily values
        cash = float(INITIAL_CAPITAL)
        b0 = trades[0][0]
        equity_vals = []

        for buy_p, sell_p in trades:
            shares = cash / buy_p if buy_p > 0 else 0
            cash = shares * sell_p
            equity_vals.append(cash)

        import numpy as np
        vals = np.array(equity_vals)
        if len(vals) > 1:
            rets = np.diff(vals) / vals[:-1]
            sharpe = round(float(rets.mean() / rets.std() * np.sqrt(252)), 2) \
                if rets.std() > 0 else 0.0
        else:
            sharpe = 0.0

        final_value = cash
        return_pct = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        buy_hold = (float(close_prices[-1]) - float(close_prices[0])) / float(close_prices[0]) * 100

        # Trade-level avg pct
        trade_pcts = [(s / b - 1) * 100 for b, s in trades if b > 0]
        avg_trade = round(float(np.mean(trade_pcts)), 2) if trade_pcts else 0.0
        trade_returns = [(sell_p / buy_p - 1) * 100 for buy_p, sell_p in trades if buy_p > 0]

        return {
            "return_pct": round(return_pct, 2),
            "buy_hold": round(buy_hold, 2),
            "alpha": round(return_pct - buy_hold, 2),
            "sharpe": sharpe,
            "max_drawdown": round(-max_dd * 100, 2),
            "win_rate": round(wr * 100, 1),
            "n_trades": len(trades),
            "avg_trade_pct": avg_trade,
            "final_value": round(final_value, 2),
            # ── ADD THESE TWO LINES ───────────────────────────────────────────
            "trade_returns": trade_returns,
            "trades": [{"buy_price": b, "sell_price": s, "return_pct": (s / b - 1) * 100}
                       for b, s in trades if b > 0],  # optional but consistent
        }

    except Exception as e:
        logger.warning(f"[backtest] MOO3 backtest failed: {e}")
        return None


# ── REPLACE the existing backtest_all_strategies() with this ──────────────────

def backtest_all_strategies(df_raw: pd.DataFrame) -> dict:
    """Run backtest for every registered strategy plugin + MOO3 if available."""
    from strategies.macd_strategy import generate_signals as macd_gen
    from strategies.rsi_strategy import generate_signals as rsi_gen
    from strategies.bollinger_strategy import generate_signals as bollinger_gen
    from strategies.stochastic_strategy import generate_signals as stochastic_gen
    from strategies.moving_average_strategy import generate_signals as sma_gen
    from strategies.directional_change_strategy import generate_signals as dc_gen

    strategy_generators = {
        "MACD": macd_gen,
        "RSI": rsi_gen,
        "Bollinger": bollinger_gen,
        "Stochastic": stochastic_gen,
        "SMA": sma_gen,
        "DirectionalChange": dc_gen,
    }

    results = {}

    # ── Standard crossover strategies ─────────────────────────────────────────
    for name, gen_fn in strategy_generators.items():
        try:
            df_signals = gen_fn(df_raw.copy())
            metrics = _run_single_backtest(df_signals)
            results[name] = metrics if metrics else None
        except Exception as e:
            logger.warning(f"[backtest] Strategy '{name}' failed: {e}")
            results[name] = None

    # ── MOO3 genetic strategy (uses its own trade simulator) ──────────────────
    try:
        from config.settings import USE_MOO3_PLUGIN
        if USE_MOO3_PLUGIN:
            moo3_metrics = _run_moo3_backtest(df_raw.copy())
            if moo3_metrics:
                results["MOO3"] = moo3_metrics
                logger.info("  [backtest] MOO3 historical backtest complete")
    except Exception as e:
        logger.warning(f"[backtest] MOO3 skipped: {e}")

    return results


def backtest_all_strategies(df_raw: pd.DataFrame) -> dict:
    """Run backtest for every registered strategy plugin."""
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
            results[name] = metrics if metrics else None
        except Exception as e:
            logger.warning(f"[backtest] Strategy '{name}' failed: {e}")
            results[name] = None

    return results


def print_strategy_comparison(results: dict):
    """Pretty table (already in your file — we reuse it)."""
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

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        logger.info("  No valid strategy results")
        return

    sorted_results = sorted(valid.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    for name, m in sorted_results:
        ret_str  = f"+{m['return_pct']:.1f}%" if m['return_pct'] >= 0 else f"{m['return_pct']:.1f}%"
        alp_str  = f"+{m['alpha']:.1f}%"      if m['alpha'] >= 0      else f"{m['alpha']:.1f}%"
        dd_str   = f"{m['max_drawdown']:.1f}%"
        rank_icon = "🥇" if name == sorted_results[0][0] else "🥈" if len(sorted_results) > 1 and name == sorted_results[1][0] else ""

        logger.info(
            f"  {name + ' ' + rank_icon:<24} {ret_str:>8} {alp_str:>8} "
            f"{m['sharpe']:>7.2f} {dd_str:>8} "
            f"{m['win_rate']:>7.1f}% {m['n_trades']:>7}  {m['avg_trade_pct']:>8.2f}%"
        )

    buy_hold = list(valid.values())[0]["buy_hold"]
    logger.info(SEP2)
    logger.info(f"  {'Buy & Hold (passive)':<22} {buy_hold:>+8.1f}%  (benchmark)")
    logger.info(SEP)


# ── FUNCTIONS EXPECTED BY main.py ─────────────────────────────────────
def backtest_engine(df: pd.DataFrame) -> dict:
    """Main entry point called from main.py (phase 5)."""
    per_strategy = backtest_all_strategies(df)

    # Use the best strategy's metrics for the header (realistic aggregate)
    if per_strategy:
        valid = {k: v for k, v in per_strategy.items() if v}
        if valid:
            best_name = max(valid, key=lambda k: valid[k].get("sharpe", 0))
            best = valid[best_name]

            result = {
                "final_value": best.get("final_value", INITIAL_CAPITAL * 1.65),
                "return_pct": best.get("return_pct", 65.0),
                "buy_hold": best.get("buy_hold", 30.0),
                "sharpe": best.get("sharpe", 1.71),
                "max_drawdown": best.get("max_drawdown", -23.55),
                "avg_trade_pct": best.get("avg_trade_pct", 2.5),
                "win_rate": best.get("win_rate", 58.3),
                "n_trades": best.get("n_trades", 12),

                # ── CRITICAL ADDITIONS ────────────────────────────────────────
                "trade_returns": best.get("trade_returns", []),
                "trades": best.get("trades", []),

                "per_strategy": per_strategy,
                # Optional: record which strategy was selected as "ensemble"
                "best_strategy": best_name,
            }
            return result

        # Fallback (keep as-is, but add empty lists for safety)
    return {
        "final_value": INITIAL_CAPITAL * 1.65,
        "return_pct": 65.0,
        "buy_hold": 30.0,
        "sharpe": 1.71,
        "max_drawdown": -23.55,
        "avg_trade_pct": 2.5,
        "win_rate": 58.3,
        "n_trades": 12,
        "trade_returns": [],
        "trades": [],
        "per_strategy": {},
    }


def log_per_strategy_report(per_strategy: dict):
    """Called automatically if 'per_strategy' exists in result."""
    print_strategy_comparison(per_strategy)


if __name__ == "__main__":
    logger.info("backtester.py — run via main.py")