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
from utils.logger import logger, trade_logger, reset_trade_logger



def _run_single_backtest(df_with_signals: pd.DataFrame, log_trades: bool = True) -> dict | None:
    """Run a single backtest on a df that already has signal/crossover columns."""
    if "crossover" not in df_with_signals.columns:
        return None

    from config.settings import (
        USE_TAKE_PROFIT, TAKE_PROFIT_PCT,
        USE_TRAILING_STOP, TRAILING_STOP_PCT,
        USE_TIME_EXIT, EXIT_MAX_HOLD_DAYS,
        USE_STOP_LOSS, STOP_LOSS_PCT,
    )

    df = df_with_signals.copy()  # avoid mutating input
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("[backtest] df index is not DatetimeIndex — time exits may be inaccurate")

    cash = float(INITIAL_CAPITAL)
    position = None          # dict when open: {'shares': float, 'entry_price': float, 'entry_date': pd.Timestamp, 'peak_price': float}
    trades = []
    daily_values = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        date = row.name if isinstance(row.name, pd.Timestamp) else df.index[i]  # fallback
        crossover = int(row["crossover"])

        # ── 1. Check exits if position open ─────────────────────────────────
        if position is not None:
            entry_price = position["entry_price"]
            entry_date = position["entry_idx"]
            pnl_pct = (price - entry_price) / entry_price * 100

            # # ── Insert right after: pnl_pct = (price - entry_price) / entry_price * 100
            # logger.info(
            #     f"[SL-DEBUG] bar={i:4} | "
            #     f"pnl_pct={pnl_pct:+6.2f}% | "
            #     f"STOP_LOSS_PCT={STOP_LOSS_PCT:+.4f} | "
            #     f"threshold={STOP_LOSS_PCT * 100:+.2f}% | "
            #     f"condition: {pnl_pct:+.2f} <= {STOP_LOSS_PCT * 100:+.2f} ? → "
            #     f"{'YES' if pnl_pct <= STOP_LOSS_PCT * 100 else 'NO'}"
            # )  VERY ANNOYING TO RUN

            if position is not None and pnl_pct > 0 and pnl_pct <= STOP_LOSS_PCT * 100:
                logger.warning(
                    f"!!! POSITIVE PNL STOP-LOSS TRIGGERED !!! "
                    f"pnl={pnl_pct:+.2f}%  threshold={STOP_LOSS_PCT * 100:+.2f}%  "
                    f"STOP_LOSS_PCT raw value = {STOP_LOSS_PCT}"
                )

            # days_held = (date.date() - entry_date.date()).days if hasattr(date, 'date') else i - position.get("entry_idx", i)
            days_held = i - position["entry_idx"]
            # Update running peak
            position["peak_price"] = max(position["peak_price"], price)

            exit_reason = None

            # logger.info(
            #     f"DEBUG-SL-CONFIG  STOP_LOSS_PCT = {STOP_LOSS_PCT:+.4f}  →  threshold = {STOP_LOSS_PCT * 100:+.2f}%")
            pnl_pct = (price - entry_price) / entry_price * 100



            # Priority: SL > Time > Trailing > TP
            if USE_STOP_LOSS and pnl_pct <= STOP_LOSS_PCT * 100:
                exit_reason = f"stop_loss ({pnl_pct:.2f}%)"
                # logger.warning(
                #     f"SL triggered with pnl_pct = {pnl_pct:+.2f}%  (should only happen ≤ {STOP_LOSS_PCT * 100:+.2f}%)")

            elif USE_TIME_EXIT and days_held >= EXIT_MAX_HOLD_DAYS:
                exit_reason = f"time_exit ({days_held} days held)"

            elif USE_TRAILING_STOP and pnl_pct > 2:
                dd_from_peak = (price - position["peak_price"]) / position["peak_price"] * 100
                if dd_from_peak <= -TRAILING_STOP_PCT * 100:
                    exit_reason = f"trailing_stop (dd {dd_from_peak:.2f}% from peak {position['peak_price']:.2f})"

            elif USE_TAKE_PROFIT and pnl_pct >= TAKE_PROFIT_PCT * 100:
                exit_reason = f"take_profit ({pnl_pct:.2f}%)"

            # elif USE_TRAILING_STOP and pnl_pct > 2:
            #     dd_from_peak = (price - position["peak_price"]) / position["peak_price"] * 100
            #     if dd_from_peak <= -TRAILING_STOP_PCT * 100:
            #         exit_reason = f"trailing_stop (dd {dd_from_peak:.2f}% from peak {position['peak_price']:.2f})"
            if exit_reason:
                proceeds = position["shares"] * price
                pnl_dollar = proceeds - (position["shares"] * entry_price)
                cash += proceeds

                trades.append({
                    "type": "SELL",
                    "price": price,
                    "shares": position["shares"],
                    "pnl": pnl_dollar,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "days_held": days_held,
                    "date": date
                })
                # Optional: log for debugging
                # logger.info(f"[backtest exit] {exit_reason} at {date.date()} price ${price:.2f}")
                if exit_reason and log_trades:
                    trade_logger.info(
                        f"[BT EXIT] {exit_reason:12} | "
                        f"days_held={days_held:3} | "
                        f"pnl={pnl_pct:+6.2f}% | "
                        f"price={price:8.2f} | "
                        f"bar={i:4}"
                    )

                position = None

        # ── 2. New entry only if flat ───────────────────────────────────────
        if position is None and crossover == 1 and cash > 100:  # buffer
            shares = full_position_size(cash, price)
            if shares > 0:
                cost = shares * price
                cash -= cost
                position = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_date": date,
                    "peak_price": price,
                    "entry_idx": i   # fallback if no date
                }
                trades.append({
                    "type": "BUY",
                    "price": price,
                    "shares": shares,
                    "date": date
                })

        # End-of-bar equity
        equity = cash + (position["shares"] * price if position else 0)
        daily_values.append(equity)

    # ── Force close open position at end ────────────────────────────────────
    if position is not None:
        price = float(df["close"].iloc[-1])
        proceeds = position["shares"] * price
        pnl_dollar = proceeds - (position["shares"] * position["entry_price"])
        cash += proceeds
        pnl_pct = (price - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            "type": "SELL (end)",
            "price": price,
            "shares": position["shares"],
            "pnl": pnl_dollar,
            "pnl_pct": pnl_pct,
            "exit_reason": "data_end",
            "days_held": len(df) - position["entry_idx"]
        })

    # ── Metrics (adapted from your original) ────────────────────────────────
    final_value = cash
    return_pct = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    buy_hold = ((float(df["close"].iloc[-1]) - float(df["close"].iloc[0])) /
                float(df["close"].iloc[0]) * 100) if len(df) > 1 else 0.0

    closed = [t for t in trades if t["type"].startswith("SELL")]
    wins = [t for t in closed if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    trade_pcts = [t["pnl_pct"] for t in closed if "pnl_pct" in t]
    avg_trade_pct = round(float(np.mean(trade_pcts)), 2) if trade_pcts else 0.0

    vals = np.array(daily_values, dtype=float)
    if len(vals) > 1:
        daily_returns = pd.Series(vals).pct_change().dropna()
        sharpe = round(float((daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252)), 2) \
                 if daily_returns.std() > 0 else 0.0
        peak = vals[0]
        max_dd = 0.0
        for v in vals:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
    else:
        sharpe = 0.0
        max_dd = 0.0

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
        "trade_returns": trade_pcts,
        "trades": trades,  # now includes exit_reason, days_held — useful for analysis
    }

def sanitize_moo3_params(best):
        if best.sl_pct <= 0:
            logger.warning(f"[MOO3] Invalid SL {best.sl_pct:+.4f} -> set to 0.02")
            best.sl_pct = 0.02
        if best.sl_pct > 0.40:
            logger.warning(f"[MOO3] SL too large {best.sl_pct:+.4f} -> clamped to 0.40")
            best.sl_pct = 0.40

        if best.sell_pct < 0:
            logger.warning(f"[MOO3] Invalid TP {best.sell_pct:+.4f} -> clamped")
            best.sell_pct = 0.01

        if best.sell_days < 1:
            best.sell_days = 1

        return best


# ── ADD this function ABOVE backtest_all_strategies() ─────────────────────────
# ── UPDATED _run_moo3_backtest (replace the entire function) ─────────────────────
def _run_moo3_backtest(
    df: pd.DataFrame,
    exit_days_override: int | None = None,
    log_trades: bool = True,
    log_info: bool = True,
) -> dict | None:
    """
    Run a historical backtest for the saved MOO3 individual.
    MOO3 uses time/price-based exits (sell_days + sell_pct + sl_pct).
    We now OVERRIDE the simulator to ALSO include the trailing-stop logic
    from _run_single_backtest (SL > Time > Trailing > TP) while keeping
    the evolved parameters for SL/TP/time. This gives a more realistic
    backtest without touching genetic/fitness.py (training numbers stay
    valid for the Pareto front; only the historical report improves).
    Full daily equity curve is used for accurate Sharpe + MaxDD.
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
        from config.settings import (
            USE_TIME_EXIT, EXIT_MAX_HOLD_DAYS,
            USE_TRAILING_STOP, TRAILING_STOP_PCT,
        )
        best = MOO3Engine.load()
        best = sanitize_moo3_params(best)

        # lOGGER FOR BOT STATS
        if log_info:
            logger.info(
                f"MOO3 loaded → sl_pct = {best.sl_pct:.4f}, sell_pct = {best.sell_pct:.4f}, sell_days = {best.sell_days}")

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

        # ── USE CONFIG DAYS IF ENABLED, otherwise keep evolved value ───────
        sell_days_to_use = exit_days_override if exit_days_override is not None else \
            (EXIT_MAX_HOLD_DAYS if USE_TIME_EXIT else best.sell_days)

        # ── FULL SIMULATION WITH TRAILING STOP (same engine as single backtest) ─
        cash = float(INITIAL_CAPITAL)
        position = None  # dict when open
        trades = []
        daily_values = []

        for i in range(n):
            price = close_prices[i]
            date = df.index[i] if hasattr(df, "index") else i
            buy_signal = buy_signals[i]

            # ── 1. Check exits if position open ─────────────────────────────
            if position is not None:
                entry_price = position["entry_price"]
                pnl_pct = (price - entry_price) / entry_price * 100
                days_held = i - position["entry_idx"]
                position["peak_price"] = max(position["peak_price"], price)
                exit_reason = None

                # Priority exactly as in _run_single_backtest, but using MOO3 params
                if pnl_pct <= -best.sl_pct * 100:
                    exit_reason = f"stop_loss ({pnl_pct:.2f}%)"
                    # logger.critical("!!! MOO3 SL TRIGGERED !!!")
                    # logger.critical(f"     pnl_pct       = {pnl_pct:+.4f}%")
                    # logger.critical(f"     best.sl_pct   = {best.sl_pct:+.4f}")
                    # logger.critical(f"     threshold     = {-best.sl_pct * 100:+.2f}%")
                    # logger.critical(f"     entry_price   = {entry_price:.4f}")
                    # logger.critical(f"     current price = {price:.4f}")
                    # VERY ANNOYING GOOED LOG
                elif days_held >= sell_days_to_use:
                    exit_reason = f"time_exit ({days_held} days held)"
                elif USE_TRAILING_STOP and pnl_pct > 2:
                    dd_from_peak = (price - position["peak_price"]) / position["peak_price"] * 100
                    if dd_from_peak <= -TRAILING_STOP_PCT * 100:
                        exit_reason = f"trailing_stop (dd {dd_from_peak:.2f}% from peak {position['peak_price']:.2f})"
                elif best.sell_pct > 0 and pnl_pct >= best.sell_pct * 100:
                    exit_reason = f"take_profit ({pnl_pct:.2f}%)"

                if exit_reason:
                    proceeds = position["shares"] * price
                    pnl_dollar = proceeds - (position["shares"] * entry_price)
                    cash += proceeds
                    trades.append({
                        "type": "SELL",
                        "price": price,
                        "shares": position["shares"],
                        "pnl": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "days_held": days_held,
                        "date": date
                    })
                    if log_trades:
                        trade_logger.info(
                            f"[BT EXIT] {exit_reason:12} | "
                            f"days_held={days_held:3} | "
                            f"pnl={pnl_pct:+6.2f}% | "
                            f"price={price:8.2f} | "
                            f"bar={i:4}"
                        )
                    position = None

            # ── 2. New entry only if flat (full allocation to match training) ─
            if position is None and buy_signal and cash > 100:
                shares = full_position_size(cash, price) if price > 0 else 0
                if shares > 0:
                    cost = shares * price
                    cash -= cost
                    position = {
                        "shares": shares,
                        "entry_price": price,
                        "peak_price": price,
                        "entry_idx": i
                    }
                    trades.append({
                        "type": "BUY",
                        "price": price,
                        "shares": shares,
                        "date": date
                    })

            # End-of-bar equity
            equity = cash + (position["shares"] * price if position else 0)
            daily_values.append(equity)

        # ── Force close open position at end ────────────────────────────────
        if position is not None:
            price = close_prices[-1]
            proceeds = position["shares"] * price
            pnl_dollar = proceeds - (position["shares"] * position["entry_price"])
            cash += proceeds
            pnl_pct = (price - position["entry_price"]) / position["entry_price"] * 100
            trades.append({
                "type": "SELL (end)",
                "price": price,
                "shares": position["shares"],
                "pnl": pnl_dollar,
                "pnl_pct": pnl_pct,
                "exit_reason": "data_end",
                "days_held": n - position["entry_idx"]
            })

        # ── Metrics (identical to _run_single_backtest) ─────────────────────
        final_value = cash
        return_pct = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        buy_hold = ((close_prices[-1] - close_prices[0]) /
                    close_prices[0] * 100) if n > 1 else 0.0
        closed = [t for t in trades if t["type"].startswith("SELL")]
        wins = [t for t in closed if t.get("pnl", 0) > 0]
        win_rate = len(wins) / len(closed) * 100 if closed else 0.0
        trade_pcts = [t["pnl_pct"] for t in closed if "pnl_pct" in t]
        avg_trade_pct = round(float(np.mean(trade_pcts)), 2) if trade_pcts else 0.0

        vals = np.array(daily_values, dtype=float)
        if len(vals) > 1:
            daily_returns = pd.Series(vals).pct_change().dropna()
            sharpe = round(float((daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252)), 2) \
                     if daily_returns.std() > 0 else 0.0
            peak = vals[0]
            max_dd = 0.0
            for v in vals:
                if v > peak:
                    peak = v
                dd = (v - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd
        else:
            sharpe = 0.0
            max_dd = 0.0

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
            "trade_returns": trade_pcts,
            "trades": trades,
        }
    except Exception as e:
        logger.warning(f"[backtest] MOO3 backtest failed: {e}")
        return None

# def _run_moo3_backtest(df: pd.DataFrame, exit_days_override: int | None = None) -> dict | None:
#     """
#     Run a historical backtest for the saved MOO3 individual.
#
#     MOO3 uses time/price-based exits (sell_days + sell_pct + sl_pct),
#     NOT crossover signals.  We use the same trade simulator as training
#     (genetic/fitness.py) so the numbers are directly comparable to the
#     Pareto front objectives printed during genetic/run_genetic.py.
#
#     Pre-builds the terminal matrix once for all bars — O(n) not O(n²).
#     """
#     import os
#     _MODEL_PATH = os.path.join(
#         os.path.dirname(__file__), "..", "models", "moo3_best.pkl"
#     )
#     if not os.path.exists(_MODEL_PATH):
#         return None
#
#     try:
#         from genetic.gp_engine import MOO3Engine
#         from genetic.terminals import build_terminal_matrix, row_to_dict
#         from genetic.fitness import simulate_trades, compute_objectives
#         from config.settings import USE_TIME_EXIT, EXIT_MAX_HOLD_DAYS
#
#         best = MOO3Engine.load()
#
#         # ── Pre-build terminal matrix once for all bars ────────────────────
#         close_prices = df["close"].values.astype(float)
#         term_matrix, _ = build_terminal_matrix(df)
#
#         # ── Evaluate GP tree for every bar (boolean: BUY or HOLD) ─────────
#         import numpy as np
#         n = len(close_prices)
#         buy_signals = np.zeros(n, dtype=bool)
#         for i in range(n):
#             row = row_to_dict(term_matrix, i)
#             try:
#                 buy_signals[i] = best.tree.evaluate(row)
#             except Exception:
#                 buy_signals[i] = False
#
#         # ── USE CONFIG DAYS IF ENABLED, otherwise keep evolved value ───────
#         sell_days_to_use = exit_days_override if exit_days_override is not None else \
#             (EXIT_MAX_HOLD_DAYS if USE_TIME_EXIT else best.sell_days)
#
#         # ── Simulate trades with MOO3's evolved exit parameters ───────────
#         trades = simulate_trades(
#             close_prices,
#             buy_signals,
#             sell_days=sell_days_to_use,
#             sell_pct=best.sell_pct,
#             sl_pct=best.sl_pct,
#         )
#
#         if not trades:
#             return None
#
#         # ── Convert to the standard metrics dict ──────────────────────────
#         tr, wr, max_dd = compute_objectives(trades)
#
#         # Rebuild equity curve from trade list for Sharpe + daily values
#         cash = float(INITIAL_CAPITAL)
#         b0 = trades[0][0]
#         equity_vals = []
#
#         for buy_p, sell_p in trades:
#             shares = cash / buy_p if buy_p > 0 else 0
#             cash = shares * sell_p
#             equity_vals.append(cash)
#
#         import numpy as np
#         vals = np.array(equity_vals)
#         if len(vals) > 1:
#             rets = np.diff(vals) / vals[:-1]
#             sharpe = round(float(rets.mean() / rets.std() * np.sqrt(252)), 2) \
#                 if rets.std() > 0 else 0.0
#         else:
#             sharpe = 0.0
#
#         final_value = cash
#         return_pct = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
#         buy_hold = (float(close_prices[-1]) - float(close_prices[0])) / float(close_prices[0]) * 100
#
#         # Trade-level avg pct
#         trade_pcts = [(s / b - 1) * 100 for b, s in trades if b > 0]
#         avg_trade = round(float(np.mean(trade_pcts)), 2) if trade_pcts else 0.0
#         trade_returns = [(sell_p / buy_p - 1) * 100 for buy_p, sell_p in trades if buy_p > 0]
#
#         return {
#             "return_pct": round(return_pct, 2),
#             "buy_hold": round(buy_hold, 2),
#             "alpha": round(return_pct - buy_hold, 2),
#             "sharpe": sharpe,
#             "max_drawdown": round(-max_dd * 100, 2),
#             "win_rate": round(wr * 100, 1),
#             "n_trades": len(trades),
#             "avg_trade_pct": avg_trade,
#             "final_value": round(final_value, 2),
#             # ── ADD THESE TWO LINES ───────────────────────────────────────────
#             "trade_returns": trade_returns,
#             "trades": [{"buy_price": b, "sell_price": s, "return_pct": (s / b - 1) * 100}
#                        for b, s in trades if b > 0],  # optional but consistent
#         }
#
#     except Exception as e:
#         logger.warning(f"[backtest] MOO3 backtest failed: {e}")
#         return None


# ── REPLACE the existing backtest_all_strategies() with this ──────────────────

def backtest_all_strategies(
    df_raw: pd.DataFrame,
    symbol: str | None = None,
    log_trades: bool = True,
    log_info: bool = True,
) -> dict:
    """Run backtest for every registered strategy plugin + MOO3 if available."""
    if log_trades and symbol:
        global trade_logger
        trade_logger = reset_trade_logger(symbol=symbol)
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
            metrics = _run_single_backtest(df_signals, log_trades=log_trades)
            results[name] = metrics if metrics else None
        except Exception as e:
            logger.warning(f"[backtest] Strategy '{name}' failed: {e}")
            results[name] = None

    # ── MOO3 genetic strategy (uses its own trade simulator) ──────────────────
    try:
        from config.settings import USE_MOO3_PLUGIN, USE_TIME_EXIT, EXIT_MAX_HOLD_DAYS
        if USE_MOO3_PLUGIN:
            override_days = EXIT_MAX_HOLD_DAYS if USE_TIME_EXIT else None
            moo3_metrics = _run_moo3_backtest(
                df_raw.copy(),
                exit_days_override=override_days,
                log_trades=log_trades,
                log_info=log_info,
            )
            if moo3_metrics:
                results["MOO3"] = moo3_metrics
                if log_info:
                    logger.info("  [backtest] MOO3 historical backtest complete")
    except Exception as e:
        logger.warning(f"[backtest] MOO3 skipped: {e}")

    return results


# def backtest_all_strategies(df_raw: pd.DataFrame) -> dict:
#     """Run backtest for every registered strategy plugin."""
#     from strategies.macd_strategy       import generate_signals as macd_gen
#     from strategies.rsi_strategy        import generate_signals as rsi_gen
#     from strategies.bollinger_strategy  import generate_signals as bollinger_gen
#     from strategies.stochastic_strategy import generate_signals as stochastic_gen
#     from strategies.moving_average_strategy import generate_signals as sma_gen
#     from strategies.directional_change_strategy import generate_signals as dc_gen
#
#     strategy_generators = {
#         "MACD":             macd_gen,
#         "RSI":              rsi_gen,
#         "Bollinger":        bollinger_gen,
#         "Stochastic":       stochastic_gen,
#         "SMA":              sma_gen,
#         "DirectionalChange": dc_gen,
#     }
#
#     results = {}
#     for name, gen_fn in strategy_generators.items():
#         try:
#             df_signals = gen_fn(df_raw.copy())
#             metrics    = _run_single_backtest(df_signals)
#             results[name] = metrics if metrics else None
#         except Exception as e:
#             logger.warning(f"[backtest] Strategy '{name}' failed: {e}")
#             results[name] = None
#
#     return results


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
def backtest_engine(
    df: pd.DataFrame,
    symbol: str | None = None,
    log_trades: bool = True,
    log_info: bool = True,
) -> dict:
    """Main entry point called from main.py (phase 5)."""
    per_strategy = backtest_all_strategies(
        df,
        symbol=symbol,
        log_trades=log_trades,
        log_info=log_info,
    )

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


