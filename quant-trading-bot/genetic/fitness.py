"""
genetic/fitness.py — Trade Simulator & Fitness Evaluator
==========================================================
Simulates the trading strategy encoded by a GP tree and computes
the three fitness objectives for NSGA-II.

Sell rule (Long et al. 2026, Section 4)
-----------------------------------------
  "Our strategy for selling an asset is to sell either after n days
   have already passed from the initial purchase, or when a price
   increase of r% has occurred, whichever comes first."

  We extend this with a stop-loss at -sl_pct to limit downside,
  which is consistent with the paper's transaction-cost model.

Objectives
----------
  f1 — Total Return      (MAXIMISE) — overall % gain on first buy price
  f2 — Win Rate          (MAXIMISE) — fraction of profitable trades
  f3 — Max Drawdown      (MINIMISE) — worst peak-to-trough equity decline

Note: f3 replaces "expected rate of return" from the original paper to
match the user's stated preference (win rate + drawdown minimisation).

Performance
-----------
  The inner trade loop is written in pure Python but is fast enough for
  P=50 population × N=50 generations on 500-bar datasets (~2-5 seconds
  single-threaded). For larger datasets, use multiprocessing (gp_engine.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from genetic.gp_tree import Node
from genetic.terminals import TERMINAL_NAMES, build_terminal_matrix, row_to_dict

# ── Constants ─────────────────────────────────────────────────────────────────

TRANSACTION_COST = 0.001   # 0.1% per side (10 bps) — paper default
DEFAULT_SELL_DAYS = 15     # max holding period (bars)
DEFAULT_SELL_PCT  = 0.05   # take-profit at +5%
DEFAULT_SL_PCT    = 0.05   # stop-loss at -5%

# Penalty returned when an individual produces zero trades
# (Pareto dominance will push zero-trade individuals to lower fronts)
NO_TRADE_PENALTY = (-0.5, 0.0, 1.0)   # (TR, WR, MaxDD)


# ── Core trade simulator ──────────────────────────────────────────────────────

def simulate_trades(
    close_prices: np.ndarray,
    buy_signals:  np.ndarray,           # bool array
    sell_days:    int   = DEFAULT_SELL_DAYS,
    sell_pct:     float = DEFAULT_SELL_PCT,
    sl_pct:       float = DEFAULT_SL_PCT,
    tx_cost:      float = TRANSACTION_COST,
) -> List[Tuple[float, float]]:
    """
    Simulate long-only trades driven by a boolean buy-signal array.

    Entry  : first bar where buy_signals[i] is True and not in position
    Exit   : sell after `sell_days` bars  OR  when gain ≥ sell_pct
             OR when loss ≤ -sl_pct  (stop-loss extension)
    Costs  : tx_cost applied on both entry and exit

    Returns list of (buy_price_after_cost, sell_price_after_cost) tuples.
    """
    n      = len(close_prices)
    trades: List[Tuple[float, float]] = []

    in_position = False
    buy_price   = 0.0
    buy_day     = 0

    for i in range(n):
        price = float(close_prices[i])

        if not in_position:
            if bool(buy_signals[i]):
                buy_price   = price * (1.0 + tx_cost)
                buy_day     = i
                in_position = True
        else:
            days_held  = i - buy_day
            pct_change = (price - buy_price) / buy_price

            exit_signal = (
                days_held >= sell_days          # time-based exit
                or pct_change >= sell_pct       # take-profit
                or pct_change <= -sl_pct        # stop-loss
            )

            if exit_signal:
                sell_price = price * (1.0 - tx_cost)
                trades.append((buy_price, sell_price))
                in_position = False

    # Close any open position at end of dataset
    if in_position and n > 0:
        sell_price = float(close_prices[-1]) * (1.0 - tx_cost)
        trades.append((buy_price, sell_price))

    return trades


# ── Objectives from trade list ─────────────────────────────────────────────────

def compute_objectives(
    trades: List[Tuple[float, float]],
) -> Tuple[float, float, float]:
    """
    Compute the three fitness objectives from a completed trade list.

    Parameters
    ----------
    trades : list of (buy_price, sell_price) — both include transaction costs

    Returns
    -------
    (total_return, win_rate, max_drawdown)
      total_return : float — overall return relative to first buy (can be > 1)
      win_rate     : float — fraction of winning trades [0, 1]
      max_drawdown : float — worst peak-to-trough drawdown [0, 1] (higher = worse)
    """
    if not trades:
        return NO_TRADE_PENALTY

    # ── Total Return (f1) ─────────────────────────────────────────────────────
    # Paper eq. (2): TR = Σ(s_i - b_i) / b_0
    # where b_0 = buy price of the first trade
    b0     = trades[0][0]
    profit = sum(s - b for b, s in trades)
    tr     = profit / b0 if b0 > 1e-9 else 0.0

    # ── Win Rate (f2) ─────────────────────────────────────────────────────────
    wins = sum(1 for b, s in trades if s > b)
    wr   = wins / len(trades)

    # ── Max Drawdown (f3) — MINIMISE ─────────────────────────────────────────
    # Compute equity curve: compound each trade's return factor
    equity = 1.0
    peak   = 1.0
    max_dd = 0.0

    for b, s in trades:
        equity *= s / b if b > 1e-9 else 1.0
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak if peak > 1e-9 else 0.0
        max_dd  = max(max_dd, dd)

    return tr, wr, max_dd


# ── Full individual evaluator ──────────────────────────────────────────────────

def evaluate_individual(
    tree:        Node,
    close_prices: np.ndarray,
    term_matrix:  np.ndarray,          # (n_bars, n_terminals) — from build_terminal_matrix
    sell_days:    int   = DEFAULT_SELL_DAYS,
    sell_pct:     float = DEFAULT_SELL_PCT,
    sl_pct:       float = DEFAULT_SL_PCT,
) -> Tuple[float, float, float]:
    """
    Full pipeline: tree evaluation → trade simulation → objectives.

    Parameters
    ----------
    tree        : evolved GP boolean expression tree
    close_prices: 1-D array of closing prices
    term_matrix : pre-built terminal matrix (terminals.build_terminal_matrix)
    sell_days, sell_pct, sl_pct : exit parameters (also evolved per-individual)

    Returns
    -------
    (total_return, win_rate, max_drawdown)
    """
    n = len(close_prices)
    if n < max(sell_days + 5, 30):
        return NO_TRADE_PENALTY

    # Evaluate tree for each bar
    buy_signals = np.zeros(n, dtype=bool)
    for i in range(n):
        row_dict = row_to_dict(term_matrix, i)
        try:
            buy_signals[i] = tree.evaluate(row_dict)
        except Exception:
            buy_signals[i] = False

    trades = simulate_trades(close_prices, buy_signals, sell_days, sell_pct, sl_pct)
    return compute_objectives(trades)


# ── Full individual evaluator + trade count ───────────────────────────────────

def evaluate_individual_with_trades(
    tree:        Node,
    close_prices: np.ndarray,
    term_matrix:  np.ndarray,          # (n_bars, n_terminals) — from build_terminal_matrix
    sell_days:    int   = DEFAULT_SELL_DAYS,
    sell_pct:     float = DEFAULT_SELL_PCT,
    sl_pct:       float = DEFAULT_SL_PCT,
) -> Tuple[Tuple[float, float, float], int]:
    """
    Full pipeline with trade count: tree evaluation → trade simulation → objectives.

    Returns
    -------
    (objectives, trade_count)
    """
    n = len(close_prices)
    if n < max(sell_days + 5, 30):
        return NO_TRADE_PENALTY, 0

    buy_signals = np.zeros(n, dtype=bool)
    for i in range(n):
        row_dict = row_to_dict(term_matrix, i)
        try:
            buy_signals[i] = tree.evaluate(row_dict)
        except Exception:
            buy_signals[i] = False

    trades = simulate_trades(close_prices, buy_signals, sell_days, sell_pct, sl_pct)
    return compute_objectives(trades), len(trades)

# ── Batch evaluator for multiprocessing ──────────────────────────────────────

def evaluate_population_batch(
    args_list,
) -> List[Tuple[float, float, float]]:
    """
    Evaluate a list of (tree, close, term_matrix, sell_days, sell_pct, sl_pct)
    tuples.  Designed for use with multiprocessing.Pool.map().
    """
    return [evaluate_individual(*args) for args in args_list]
