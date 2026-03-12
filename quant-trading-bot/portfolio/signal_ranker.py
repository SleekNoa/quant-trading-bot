"""
Signal Ranker & Capital Allocator
===================================
Ranks signals from a multi-ticker sweep and allocates capital across
the top-N candidates.

Composite Score
---------------
    score = Sharpe(norm) × 0.6  +  prob_signal × 0.4

where:
    Sharpe(norm) = min(max(sharpe, 0), 3) / 3   → normalised to 0 – 1
    prob_signal  = P(up) if BUY, 1 − P(up) if SELL   → already 0 – 1

The 60/40 weighting reflects the principle that risk-adjusted return
quality (Sharpe) should dominate over probabilistic edge (prob).

Capital Allocation Methods
--------------------------
    equal            : split evenly across ranked tickers  (default)
    score_weighted   : allocate proportional to composite score
    sharpe_weighted  : allocate proportional to Sharpe ratio

All methods enforce a per-ticker cap (default 40%) to prevent over-
concentration.

Usage
-----
    from portfolio.signal_ranker import (
        rank_signals, allocate_capital, print_ranking_report
    )

    # ticker_results = {ticker: engine_result_dict, ...}
    ranked    = rank_signals(ticker_results, top_n=3)
    allocated = allocate_capital(ranked, total_capital=10000)
    print_ranking_report(allocated, total_capital=10000)
"""

from __future__ import annotations

from typing import Optional

SEP2 = "-" * 70


# ── Signal ranking ────────────────────────────────────────────────────────────

def rank_signals(
    ticker_results: dict,
    top_n:          int = 3,
    require_signal: int = 1,   # 1=BUY only  |  -1=SELL only  |  0=any non-HOLD
) -> list[dict]:
    """
    Score and rank tickers by composite Sharpe + probability signal.

    Parameters
    ----------
    ticker_results : dict[ticker → result_dict]
        Each result_dict should contain:
            decision   : str    "BUY" | "SELL" | "HOLD"
            sharpe     : float  backtest Sharpe ratio
            prob_up    : float  P(up) from LogisticProbabilityModel  (0.0 – 1.0)
            return_pct : float  backtest total return %
            price      : float  latest close price
        Any missing keys default to 0 / neutral.

    top_n          : maximum number of ranked signals to return
    require_signal : direction filter
                     1  = return BUY  signals only  (default)
                    -1  = return SELL signals only
                     0  = return any signal (exclude HOLD)

    Returns
    -------
    list[dict]  sorted by score descending, length ≤ top_n
    """
    ranked: list[dict] = []

    for ticker, r in ticker_results.items():
        if r is None:
            continue

        decision = r.get("decision", "HOLD")

        # Direction filter
        if require_signal == 1 and decision != "BUY":
            continue
        if require_signal == -1 and decision != "SELL":
            continue
        if require_signal == 0 and decision == "HOLD":
            continue

        sharpe   = float(r.get("sharpe",     0.0))
        prob_up  = float(r.get("prob_up",    0.5))
        ret      = float(r.get("return_pct", 0.0))
        price    = float(r.get("price",      0.0))

        # ── Composite score ────────────────────────────────────────────────────
        sharpe_norm  = min(max(sharpe, 0.0), 3.0) / 3.0   # normalise Sharpe → 0–1
        prob_signal  = prob_up if decision == "BUY" else (1.0 - prob_up)
        score        = sharpe_norm * 0.6 + prob_signal * 0.4

        ranked.append({
            "ticker":     ticker,
            "decision":   decision,
            "score":      round(score,   4),
            "sharpe":     round(sharpe,  3),
            "prob_up":    round(prob_up, 3),
            "return_pct": round(ret,     2),
            "price":      price,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]


# ── Capital allocation ────────────────────────────────────────────────────────

def allocate_capital(
    ranked:        list[dict],
    total_capital: float,
    method:        str   = "equal",
    max_pct:       float = 0.40,    # max 40% of capital to any single ticker
) -> list[dict]:
    """
    Add ``allocated_dollars`` and ``allocation_pct`` to each ranked entry.

    Parameters
    ----------
    ranked        : output of rank_signals() — modified in-place
    total_capital : cash available (e.g. account.equity)
    method        : "equal" | "score_weighted" | "sharpe_weighted"
    max_pct       : maximum fraction to any single ticker

    Returns
    -------
    Same list with allocation fields added.
    """
    if not ranked:
        return ranked

    n = len(ranked)

    if method == "equal":
        raw_fracs = [1.0 / n] * n

    elif method == "score_weighted":
        scores    = [r["score"] for r in ranked]
        total_s   = sum(scores) or 1e-9
        raw_fracs = [s / total_s for s in scores]

    elif method == "sharpe_weighted":
        sharpes   = [max(r["sharpe"], 0.0) for r in ranked]
        total_sh  = sum(sharpes) or 1e-9
        raw_fracs = [s / total_sh for s in sharpes]

    else:
        raise ValueError(f"Unknown allocation method: {method!r}.  Use 'equal', 'score_weighted', or 'sharpe_weighted'.")

    # Apply per-ticker cap
    fracs   = [min(f, max_pct) for f in raw_fracs]
    total_f = sum(fracs) or 1e-9
    fracs   = [f / total_f for f in fracs]  # renormalise after capping

    for entry, frac in zip(ranked, fracs):
        entry["allocated_dollars"] = round(total_capital * frac, 2)
        entry["allocation_pct"]    = round(frac * 100, 1)

    return ranked


# ── Formatted report ──────────────────────────────────────────────────────────

def print_ranking_report(
    ranked:        list[dict],
    total_capital: float,
    symbol_in_use: Optional[str] = None,
) -> None:
    """Log a formatted signal ranking table."""
    from utils.logger import logger

    logger.info(SEP2)
    logger.info("  MULTI-TICKER SIGNAL RANKING")
    logger.info(
        f"  {'Rank':<5}  {'Ticker':<7}  {'Signal':<5}  "
        f"{'Score':>6}  {'Sharpe':>7}  {'P(up)':>6}  "
        f"{'OOS Ret':>8}  {'Alloc $':>10}  {'%':>5}"
    )
    logger.info(SEP2)

    for i, r in enumerate(ranked, 1):
        alloc_d  = r.get("allocated_dollars", 0.0)
        alloc_p  = r.get("allocation_pct",    0.0)
        ticker   = r["ticker"]
        tag      = " ←" if ticker == symbol_in_use else "  "

        logger.info(
            f"  {i:<5}  {ticker:<7}  {r['decision']:<5}  "
            f"{r['score']:>6.3f}  {r['sharpe']:>7.2f}  "
            f"{r['prob_up']:>6.2f}  "
            f"{r['return_pct']:>+7.1f}%  "
            f"${alloc_d:>9,.0f}  {alloc_p:>5.1f}%{tag}"
        )

    logger.info(SEP2)

    if symbol_in_use:
        logger.info(
            f"  ← = currently executing SYMBOL  "
            f"(USE_MULTI_TICKER selects the top-ranked BUY)"
        )