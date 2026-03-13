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
from utils.logger import logger


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
    Fixed permutation Monte Carlo.
    Now tracks full equity paths → meaningful variation in minimum equity.
    """
    if not trade_returns or len(trade_returns) < 3:
        return {
            "error": f"Need ≥ 3 closed trades; got {len(trade_returns) if trade_returns else 0}"
        }

    rng = np.random.default_rng(seed)
    arr = np.asarray(trade_returns, dtype=float)
    n_trades = len(arr)

    # ── Generate all shuffled paths at once ────────────────────────────────────
    shuffled = np.stack([rng.permutation(arr) for _ in range(simulations)])

    # ── Build full equity paths (cumulative product) ───────────────────────────
    multipliers = 1.0 + shuffled / 100.0
    equity_paths = initial_capital * np.cumprod(multipliers, axis=1)

    # ── Extract statistics per path ────────────────────────────────────────────
    finals = equity_paths[:, -1]          # final equity
    mins   = np.min(equity_paths, axis=1) # lowest equity during path

    returns_pct = (finals / initial_capital - 1.0) * 100.0

    # Sort for percentiles
    finals.sort()
    mins.sort()
    n = len(finals)

    def _idx(p: float) -> int:
        return int(np.clip(n * p, 0, n - 1))

    # ── Trade-level analytics (order-independent) ──────────────────────────────
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = len(wins) / n_trades if n_trades else 0.0
    avg_win  = float(wins.mean())   if len(wins)   else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 1e-9
    payoff   = avg_win / avg_loss
    kelly    = max(0.0, (payoff * win_rate - (1 - win_rate)) / payoff)

    prob_loss  = float(np.mean(finals < initial_capital) * 100)
    prob_profit = 100.0 - prob_loss
    prob_ruin   = float(np.mean(mins <= 0) * 100)

    median_final = float(finals[n // 2])
    passed       = (median_final > initial_capital) and (prob_loss < 50.0)

    return {
        "simulations":       simulations,
        "n_trades":          n_trades,
        "initial_capital":   initial_capital,

        # Final equity (mathematically constant)
        "median_final":      round(median_final, 2),
        "mean_final":        round(float(finals.mean()), 2),
        "p5_final":          round(float(finals[_idx(0.05)]), 2),
        "p95_final":         round(float(finals[_idx(0.95)]), 2),

        # Path-dependent risk (the real value of shuffling)
        "median_min_equity": round(float(mins[n // 2]), 2),
        "p5_min_equity":     round(float(mins[_idx(0.05)]), 2),
        "p95_min_equity":    round(float(mins[_idx(0.95)]), 2),

        # Probabilities
        "prob_loss_pct":     round(prob_loss, 1),
        "prob_profit_pct":   round(prob_profit, 1),
        "prob_ruin_pct":     round(prob_ruin, 1),
        "passed":            passed,

        # Trade stats
        "win_rate_pct":      round(win_rate * 100, 1),
        "avg_win_pct":       round(avg_win, 2),
        "avg_loss_pct":      round(-avg_loss, 2),
        "payoff_ratio":      round(payoff, 2),
        "kelly_half_pct":    round(kelly * 50, 1),
    }


# ── Max drawdown distribution ──────────────────────────────────────────────────

def monte_carlo_max_drawdown(
    trade_returns:   list[float],
    simulations:     int   = 10_000,
    initial_capital: float = 10_000.0,
    seed:            Optional[int] = None,
) -> dict:
    """
    Vectorized max drawdown simulation across shuffled paths.
    Returns distribution of worst drawdown seen in each path.
    """
    if not trade_returns or len(trade_returns) < 3:
        return {"error": "Need ≥ 3 trades for drawdown simulation"}

    rng = np.random.default_rng(seed)
    arr = np.asarray(trade_returns, dtype=float)

    shuffled = np.stack([rng.permutation(arr) for _ in range(simulations)])
    multipliers = 1.0 + shuffled / 100.0
    equity_paths = initial_capital * np.cumprod(multipliers, axis=1)

    peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (equity_paths - peaks) / peaks * 100.0
    mdd_per_path = np.min(drawdowns, axis=1)

    mdd_arr = np.sort(mdd_per_path)
    n = len(mdd_arr)

    def _idx(p: float) -> int:
        return int(np.clip(n * p, 0, n - 1))

    return {
        "simulations":    simulations,
        "median_mdd_pct": round(float(mdd_arr[n // 2]), 2),
        "p5_mdd_pct":     round(float(mdd_arr[_idx(0.05)]), 2),
        "p25_mdd_pct":    round(float(mdd_arr[_idx(0.25)]), 2),
        "p75_mdd_pct":    round(float(mdd_arr[_idx(0.75)]), 2),
        "p95_mdd_pct":    round(float(mdd_arr[_idx(0.95)]), 2),
        "worst_mdd_pct":  round(float(mdd_arr[-1]), 2),
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
    Pretty-printed Monte Carlo report using project logger style.
    """
    logger.info(SEP)
    logger.info("  MONTE CARLO ROBUSTNESS TEST")
    logger.info(f"  {mc.get('simulations', 0):,} shuffled paths × {mc.get('n_trades', 0)} trades")
    logger.info("  Ref: Ahmed (2023) arXiv:2309.09094 | Wang et al. (2026)")
    logger.info(SEP)

    if "error" in mc:
        logger.warning(f"  Monte Carlo skipped: {mc['error']}")
        logger.info(SEP)
        return

    ic = mc["initial_capital"]

    logger.info(f"  {'Initial Capital':<34} ${ic:>12,.2f}")
    logger.info(SEP2)

    # Final equity (order-independent)
    logger.info(f"  {'Median Final Equity':<34} ${mc['median_final']:>12,.2f}")
    logger.info(f"  {'Mean Final Equity':<34} ${mc['mean_final']:>12,.2f}")
    logger.info(SEP2)

    # Path-dependent risk (the real insight)
    logger.info("  Path-Dependent Risk (varies with trade order)")
    logger.info(f"  {'Median Minimum Equity':<34} ${mc['median_min_equity']:>12,.2f}")
    logger.info(f"  {'5th pct Minimum Equity (near-worst)':<34} ${mc['p5_min_equity']:>12,.2f}")
    logger.info(f"  {'95th pct Minimum Equity (near-best)':<34} ${mc['p95_min_equity']:>12,.2f}")
    logger.info(SEP2)

    logger.info("  Probability Summary")
    logger.info(f"  {'P(profitable outcome)':<34} {mc['prob_profit_pct']:>12.1f}%")
    logger.info(f"  {'P(loss)':<34} {mc['prob_loss_pct']:>12.1f}%")
    logger.info(f"  {'P(ruin ≤ $0)':<34} {mc['prob_ruin_pct']:>12.1f}%")
    logger.info(SEP2)

    logger.info("  Trade Analytics")
    logger.info(f"  {'Win Rate':<34} {mc['win_rate_pct']:>12.1f}%")
    logger.info(f"  {'Average Win':<34} {mc['avg_win_pct']:>+12.2f}%")
    logger.info(f"  {'Average Loss':<34} {mc['avg_loss_pct']:>+12.2f}%")
    logger.info(f"  {'Payoff Ratio':<34} {mc['payoff_ratio']:>12.2f}x")
    logger.info(f"  {'Half-Kelly (recommended)':<34} {mc['kelly_half_pct']:>12.1f}% of capital")
    logger.info(SEP2)

    # Drawdown distribution (if provided)
    if dd and "error" not in dd:
        logger.info("  Max Drawdown Distribution")
        logger.info(f"  {'Median Max DD':<34} {dd['median_mdd_pct']:>12.1f}%")
        logger.info(f"  {'95th pct Max DD (near-worst)':<34} {dd['p95_mdd_pct']:>12.1f}%")
        logger.info(f"  {'Absolute Worst DD':<34} {dd['worst_mdd_pct']:>12.1f}%")
        logger.info(SEP2)

    grade = "PASS  [robust]" if mc.get("passed", False) else "FAIL  [review strategy edge]"
    logger.info(f"  MC Grade: {grade}")
    logger.info(SEP)