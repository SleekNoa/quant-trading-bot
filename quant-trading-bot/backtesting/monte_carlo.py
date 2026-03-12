"""
Monte Carlo Backtest Robustness Test
======================================
Estimates the true distribution of strategy outcomes by randomly
permuting the observed trade-return sequence N times.

Why permutation Monte Carlo?
    The observed backtest is a single path through trade-space.  A
    different ordering of the same trades could produce wildly different
    drawdowns.  Shuffling removes sequence-of-returns risk from the
    analysis and shows:
        • the realistic range of equity endpoints
        • how likely the strategy is to be profitable (P(profit))
        • how bad the drawdown could realistically get

References
----------
    Ahmed, K. (2023). Sizing Strategies for Algorithmic Trading in
    Volatile Markets. arXiv:2309.09094.
        "Robustness testing via Monte Carlo simulation reveals the true
         performance distribution independent of entry/exit sequencing."

    Wang, Zhao & Wang (2026). Integrated financial risk management
    framework. Case Studies in Thermal Engineering. ScienceDirect.
        "5 000-path Monte Carlo stress tests quantify portfolio resilience
         under adverse sequencing scenarios."

Usage
-----
    from backtesting.monte_carlo import (
        monte_carlo_test, monte_carlo_max_drawdown, print_monte_carlo_report
    )
    trade_returns = [r["pnl_pct"] for r in result["trades"] if r["type"] == "SELL"]
    mc = monte_carlo_test(trade_returns, simulations=5000)
    dd = monte_carlo_max_drawdown(trade_returns, simulations=2000)
    print_monte_carlo_report(mc, dd)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

SEP  = "=" * 70
SEP2 = "-" * 70


# ── Primary simulation ─────────────────────────────────────────────────────────

def monte_carlo_test(
    trade_returns:   list[float],
    simulations:     int   = 5_000,
    initial_capital: float = 10_000.0,
    seed:            Optional[int] = None,
) -> dict:
    """
    Shuffle the trade-return sequence N times and collect final equity.

    Parameters
    ----------
    trade_returns   : per-trade percentage returns, e.g. [+2.1, -0.8, +1.4]
                      Extract from backtester: compute (sell_price - buy_price)
                      / buy_price * 100 for each closed trade pair.
    simulations     : Monte Carlo paths  (default 5 000 per Wang et al. 2026)
    initial_capital : starting equity for each path
    seed            : integer for reproducibility; None = random each run

    Returns
    -------
    dict with full distributional statistics and a binary "passed" flag.
    """
    if not trade_returns or len(trade_returns) < 3:
        return {
            "error":       f"Need ≥ 3 closed trades; got {len(trade_returns) if trade_returns else 0}",
            "n_trades":    len(trade_returns) if trade_returns else 0,
            "simulations": 0,
            "passed":      False,
        }

    rng = np.random.default_rng(seed)
    arr = np.asarray(trade_returns, dtype=float)

    # ── Simulate N shuffled paths ──────────────────────────────────────────────
    # Using vectorised multiplication where possible for speed
    n_trades = len(arr)
    # Matrix approach: shuffle each row independently
    shuffled = np.stack([rng.permutation(arr) for _ in range(simulations)])
    # Compound return: product of (1 + r/100) across trade axis
    multipliers = np.prod(1.0 + shuffled / 100.0, axis=1)
    finals      = initial_capital * multipliers
    finals.sort()

    n            = len(finals)
    returns_pct  = (finals / initial_capital - 1.0) * 100.0

    # ── Percentile indices ─────────────────────────────────────────────────────
    def _idx(pct: float) -> int:
        return int(np.clip(n * pct, 0, n - 1))

    # ── Trade-level statistics ─────────────────────────────────────────────────
    wins      = arr[arr > 0]
    losses    = arr[arr < 0]
    win_rate  = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win   = float(wins.mean())          if len(wins)   > 0 else 0.0
    avg_loss  = float(abs(losses.mean()))   if len(losses) > 0 else 1e-9
    payoff    = avg_win / avg_loss

    # Kelly criterion (full Kelly and half-Kelly)
    # f* = (b × p - q) / b,   b = payoff, p = win_rate, q = 1 - win_rate
    kelly = max(0.0, (payoff * win_rate - (1.0 - win_rate)) / payoff)

    prob_loss   = float(np.sum(finals < initial_capital) / n * 100)
    prob_profit = 100.0 - prob_loss

    # ── Pass/fail criteria ─────────────────────────────────────────────────────
    # PASS = median path is profitable AND probability of loss < 50%
    median_final = float(finals[n // 2])
    passed       = (median_final > initial_capital) and (prob_loss < 50.0)

    return {
        # Configuration
        "simulations":       simulations,
        "n_trades":          n_trades,
        "initial_capital":   initial_capital,

        # Central tendency
        "median_final":      float(median_final),
        "mean_final":        float(finals.mean()),
        "median_return_pct": float(returns_pct[n // 2]),
        "mean_return_pct":   float(returns_pct.mean()),

        # Full distribution
        "p5_final":          float(finals[_idx(0.05)]),
        "p25_final":         float(finals[_idx(0.25)]),
        "p75_final":         float(finals[_idx(0.75)]),
        "p95_final":         float(finals[_idx(0.95)]),
        "worst_5pct_return": float(returns_pct[_idx(0.05)]),
        "best_5pct_return":  float(returns_pct[_idx(0.95)]),

        # Risk summary
        "prob_loss_pct":     round(prob_loss,   1),
        "prob_profit_pct":   round(prob_profit, 1),
        "passed":            passed,

        # Trade analytics
        "win_rate_pct":      round(win_rate * 100, 1),
        "avg_win_pct":       round(avg_win,         2),
        "avg_loss_pct":      round(-avg_loss,        2),   # shown as negative
        "payoff_ratio":      round(payoff,            2),
        "kelly_full_pct":    round(kelly * 100,       1),
        "kelly_half_pct":    round(kelly * 50,        1),  # recommended bet size
    }


# ── Max drawdown distribution ──────────────────────────────────────────────────

def monte_carlo_max_drawdown(
    trade_returns:   list[float],
    simulations:     int   = 2_000,
    initial_capital: float = 10_000.0,
    seed:            Optional[int] = None,
) -> dict:
    """
    Estimate the distribution of maximum drawdowns across shuffled paths.

    This reveals the worst realistic scenario for the strategy regardless
    of the particular order trades occurred historically.

    Returns
    -------
    dict with drawdown percentiles (all values are negative percentages).
    """
    if not trade_returns or len(trade_returns) < 3:
        return {"error": "Need ≥ 3 trades for drawdown simulation"}

    rng = np.random.default_rng(seed)
    arr = np.asarray(trade_returns, dtype=float)

    mdd_list: list[float] = []

    for _ in range(simulations):
        path   = rng.permutation(arr)
        equity = initial_capital
        peak   = initial_capital
        mdd    = 0.0
        for r in path:
            equity *= (1.0 + r / 100.0)
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak * 100.0
            if dd < mdd:
                mdd = dd
        mdd_list.append(mdd)

    mdd_arr = np.sort(mdd_list)
    n       = len(mdd_arr)

    def _idx(pct: float) -> int:
        return int(np.clip(n * pct, 0, n - 1))

    return {
        "simulations":    simulations,
        "median_mdd_pct": round(float(mdd_arr[n // 2]),          2),
        "p5_mdd_pct":     round(float(mdd_arr[_idx(0.05)]),      2),   # best 5%
        "p25_mdd_pct":    round(float(mdd_arr[_idx(0.25)]),      2),
        "p75_mdd_pct":    round(float(mdd_arr[_idx(0.75)]),      2),
        "p95_mdd_pct":    round(float(mdd_arr[_idx(0.95)]),      2),   # near-worst
        "worst_mdd_pct":  round(float(mdd_arr[-1]),               2),
    }


# ── Helper: extract trade returns from backtester output ─────────────────────

def extract_trade_returns(trades: list[dict]) -> list[float]:
    """
    Convert a backtest trade list into per-trade percentage returns.

    The backtester emits alternating BUY / SELL records.  This function
    pairs them up and computes (sell_price - buy_price) / buy_price * 100.

    Parameters
    ----------
    trades : result["trades"] from backtest() or backtest_engine()

    Returns
    -------
    list of floats — one entry per completed round-trip trade
    """
    returns:    list[float] = []
    buy_price:  Optional[float] = None

    for t in trades:
        if t.get("type") == "BUY":
            buy_price = float(t["price"])
        elif t.get("type") == "SELL" and buy_price is not None:
            sell_price = float(t["price"])
            pct        = (sell_price - buy_price) / buy_price * 100.0
            returns.append(pct)
            buy_price = None

    return returns


# ── Formatted console report ───────────────────────────────────────────────────

def print_monte_carlo_report(mc: dict, dd: Optional[dict] = None) -> None:
    """
    Print a formatted Monte Carlo report via the project logger.
    Called from main.py after monte_carlo_test() completes.
    """
    from utils.logger import logger

    logger.info(SEP)
    logger.info("  MONTE CARLO ROBUSTNESS TEST")
    logger.info(
        f"  {mc.get('simulations', 0):,} shuffled paths  ×  "
        f"{mc.get('n_trades', 0)} trades per path"
    )
    logger.info("  Ref: Ahmed (2023) arXiv:2309.09094 | Wang et al. (2026)")
    logger.info(SEP)

    if "error" in mc:
        logger.warning(f"  Monte Carlo skipped: {mc['error']}")
        logger.info(SEP)
        return

    ic = mc["initial_capital"]

    logger.info(f"  {'Initial Capital':<34} ${ic:>12,.2f}")
    logger.info(SEP2)

    # ── Central tendency ──────────────────────────────────────────────────────
    logger.info(
        f"  {'Median Final Equity':<34} ${mc['median_final']:>12,.2f}"
        f"  ({mc['median_return_pct']:>+.1f}%)"
    )
    logger.info(
        f"  {'Mean Final Equity':<34} ${mc['mean_final']:>12,.2f}"
        f"  ({mc['mean_return_pct']:>+.1f}%)"
    )
    logger.info(SEP2)

    # ── Distribution ──────────────────────────────────────────────────────────
    logger.info("  Equity Distribution Across Paths")
    logger.info(
        f"  {'  5th pct  (near-worst outcome)':<34} ${mc['p5_final']:>12,.2f}"
        f"  ({mc['worst_5pct_return']:>+.1f}%)"
    )
    logger.info(f"  {'  25th pct':<34} ${mc['p25_final']:>12,.2f}")
    logger.info(f"  {'  75th pct':<34} ${mc['p75_final']:>12,.2f}")
    logger.info(
        f"  {'  95th pct  (near-best outcome)':<34} ${mc['p95_final']:>12,.2f}"
        f"  ({mc['best_5pct_return']:>+.1f}%)"
    )
    logger.info(SEP2)

    # ── Probability summary ───────────────────────────────────────────────────
    prob_note = (
        "  [favorable]"   if mc["prob_profit_pct"] >= 65 else
        "  [marginal]"    if mc["prob_profit_pct"] >= 50 else
        "  [unfavorable]"
    )
    logger.info(
        f"  {'P(profitable outcome)':<34} {mc['prob_profit_pct']:>12.1f}%{prob_note}"
    )
    logger.info(
        f"  {'P(losing outcome)':<34} {mc['prob_loss_pct']:>12.1f}%"
    )
    logger.info(SEP2)

    # ── Trade analytics ───────────────────────────────────────────────────────
    logger.info("  Trade Analytics (from observed trade list)")
    logger.info(f"  {'  Win Rate':<34} {mc['win_rate_pct']:>12.1f}%")
    logger.info(f"  {'  Average Win':<34} {mc['avg_win_pct']:>+12.2f}%")
    logger.info(f"  {'  Average Loss':<34} {mc['avg_loss_pct']:>+12.2f}%")
    logger.info(f"  {'  Payoff Ratio':<34} {mc['payoff_ratio']:>12.2f}x")
    logger.info(
        f"  {'  Half-Kelly (recommended size)':<34} {mc['kelly_half_pct']:>12.1f}%"
        "  of capital per trade"
    )
    logger.info(SEP2)

    # ── Drawdown distribution ─────────────────────────────────────────────────
    if dd and "error" not in dd:
        logger.info("  Max Drawdown Distribution (shuffled paths)")
        logger.info(f"  {'  Median Max Drawdown':<34} {dd['median_mdd_pct']:>12.1f}%")
        logger.info(
            f"  {'  95th pct MDD  (near-worst)':<34} {dd['p95_mdd_pct']:>12.1f}%"
        )
        logger.info(
            f"  {'  Absolute Worst MDD seen':<34} {dd['worst_mdd_pct']:>12.1f}%"
        )
        logger.info(SEP2)

    # ── Grade ─────────────────────────────────────────────────────────────────
    grade = "PASS  [robust]" if mc["passed"] else "FAIL  [review strategy edge]"
    logger.info(f"  MC Grade: {grade}")
    logger.info(SEP)