"""
genetic/sharpe_selector.py — Modified Sharpe Ratio (mSR) Selector
===================================================================
Selects a single definitive strategy from the Pareto front using the
modified Sharpe Ratio proposed by Long et al. (2026).

Paper definition (Section 4)
------------------------------
  "We define a new aggregate metric, which effectively acts as a
   generalisation of the Sharpe ratio that is able to take into account
   total return as well as expected rate of return and risk."

  mSR = (w1 * normalise(f1) + w2 * normalise(f2)) / normalise(f3)

  where f1 = TR, f2 = E[RoR], f3 = Risk  (all normalised within front)

Our adaptation (user objectives)
----------------------------------
  f1 = Total Return    (MAXIMISE) — weight 0.40
  f2 = Win Rate        (MAXIMISE) — weight 0.30
  f3 = Max Drawdown    (MINIMISE) — used as denominator (penalty term)

  mSR_i = (w1 * n_TR_i + w2 * n_WR_i) / (w3 * n_DD_i + ε)

  where n_* = min-max normalised value within the Pareto front.

The individual maximising mSR is selected as the definitive strategy.

Usage
-----
    from genetic.sharpe_selector import select_from_pareto
    best_idx = select_from_pareto(pareto_objectives, weights=(0.4, 0.3, 0.3))
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# ── Default user-configured weights ──────────────────────────────────────────

DEFAULT_W_TR  = 0.40   # Total Return weight
DEFAULT_W_WR  = 0.30   # Win Rate weight
DEFAULT_W_DD  = 0.30   # Max Drawdown penalty weight


# ── Normalisation helper ──────────────────────────────────────────────────────

def _minmax_normalise(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalise to [0, 1]. Constant arrays return 0.5."""
    lo, hi = values.min(), values.max()
    if abs(hi - lo) < eps:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


# ── Modified Sharpe Ratio selection ──────────────────────────────────────────

def select_from_pareto(
    pareto_objectives: np.ndarray,      # (front_size, 3): TR, WR, MaxDD
    weights: Tuple[float, float, float] = (DEFAULT_W_TR, DEFAULT_W_WR, DEFAULT_W_DD),
    eps: float = 1e-6,
    dd_floor: float = 0.02,
    min_tr: float = 2.0,
    trade_counts: Optional[np.ndarray] = None,
    min_trades: int = 25,
) -> int:
    """
    Select the best individual from a Pareto front using the mSR criterion.

    Parameters
    ----------
    pareto_objectives : (front_size, 3) array
                        columns: [TotalReturn, WinRate, MaxDrawdown]
    weights           : (w_tr, w_wr, w_dd) — must sum to 1.0

    Returns
    -------
    best_idx : int — index into pareto_objectives of the selected individual
    """
    if len(pareto_objectives) == 0:
        raise ValueError("Empty Pareto front — cannot select.")
    if len(pareto_objectives) == 1:
        return 0

    w_tr, w_wr, w_dd = weights

    if trade_counts is not None and len(trade_counts) != len(pareto_objectives):
        trade_counts = None

    eligible = np.where(pareto_objectives[:, 0] >= min_tr)[0]
    if trade_counts is not None and min_trades > 0:
        eligible = eligible[trade_counts[eligible] >= min_trades]
    if eligible.size == 0:
        eligible = np.where(pareto_objectives[:, 0] >= min_tr)[0]
    if eligible.size == 0:
        eligible = np.arange(len(pareto_objectives))

    n_tr  = _minmax_normalise(pareto_objectives[eligible, 0])   # TR  → higher = better
    n_wr  = _minmax_normalise(pareto_objectives[eligible, 1])   # WR  → higher = better
    n_dd  = _minmax_normalise(pareto_objectives[eligible, 2])   # DD  → lower  = better
    n_dd = np.maximum(n_dd, dd_floor)
    n_dd_inv = 1.0 - n_dd                                # invert so higher = better

    # Numerator: weighted quality score (maximise)
    numerator = w_tr * n_tr + w_wr * n_wr

    # Denominator: risk penalty (higher drawdown = larger penalty)
    denominator = w_dd * n_dd + eps

    msr_scores = numerator / denominator
    best_local = int(np.argmax(msr_scores))
    return int(eligible[best_local])


def describe_pareto_front(
    pareto_objectives: np.ndarray,
    weights: Tuple[float, float, float] = (DEFAULT_W_TR, DEFAULT_W_WR, DEFAULT_W_DD),
    dd_floor: float = 0.02,
    min_tr: float = 2.0,
    trade_counts: Optional[np.ndarray] = None,
    min_trades: int = 25,
) -> str:
    """
    Return a formatted summary table of the Pareto front + selected individual.
    """
    if len(pareto_objectives) == 0:
        return "  [Pareto front is empty]\n"

    if trade_counts is not None and len(trade_counts) != len(pareto_objectives):
        trade_counts = None

    best_idx = select_from_pareto(
        pareto_objectives,
        weights,
        dd_floor=dd_floor,
        min_tr=min_tr,
        trade_counts=trade_counts,
        min_trades=min_trades,
    )
    header = f"  {'#':<4}  {'TR':>8}  {'WinRate':>8}  {'MaxDD':>8}"
    if trade_counts is not None:
        header += f"  {'Trades':>6}"
    header += f"  {'mSR':>8}"
    lines = [
        header,
        "  " + "-" * 60,
    ]

    # Compute mSR for display
    w_tr, w_wr, w_dd = weights
    eligible = np.where(pareto_objectives[:, 0] >= min_tr)[0]
    if trade_counts is not None and min_trades > 0:
        eligible = eligible[trade_counts[eligible] >= min_trades]
    if eligible.size == 0:
        eligible = np.where(pareto_objectives[:, 0] >= min_tr)[0]
    if eligible.size == 0:
        eligible = np.arange(len(pareto_objectives))

    n_tr  = _minmax_normalise(pareto_objectives[eligible, 0])
    n_wr  = _minmax_normalise(pareto_objectives[eligible, 1])
    n_dd  = _minmax_normalise(pareto_objectives[eligible, 2])
    n_dd = np.maximum(n_dd, dd_floor)
    denom = w_dd * n_dd + 1e-6
    msr   = (w_tr * n_tr + w_wr * n_wr) / denom
    msr_all = np.full(len(pareto_objectives), np.nan, dtype=float)
    msr_all[eligible] = msr

    for i, (obj, m) in enumerate(zip(pareto_objectives, msr_all)):
        marker = " ★" if i == best_idx else ""
        row = f"  {i:<4}  {obj[0]:>+7.3f}  {obj[1]:>7.1%}  {obj[2]:>7.1%}"
        if trade_counts is not None:
            row += f"  {int(trade_counts[i]):>6}"
        row += f"  {m:>8.3f}{marker}"
        lines.append(row)

    lines.append("  " + "-" * 60)
    lines.append(f"  Selected: #{best_idx}  (★ = best mSR)")
    return "\n".join(lines)
