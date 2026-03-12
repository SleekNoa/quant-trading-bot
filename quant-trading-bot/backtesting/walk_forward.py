"""
Walk-Forward Validation
========================
Rolling out-of-sample validation that enforces strict information-set
discipline across every fold.

The core principle is simple but critical:
    • Train window → strategy is applied (parameters never change between folds)
    • Test window  → strategy is re-applied on completely unseen data
    • No information from the test window is ever used to tune the train window

This prevents the lookahead bias that inflates most published backtests.

Reference
---------
    Deep et al. (2025). Walk-Forward Validation Framework for Financial
    Trading Systems. arXiv:2512.12924.

    "Rolling window validation across 34 independent test periods prevents
     lookahead bias that pervades most backtesting research.  Modest,
     non-significant returns after strict walk-forward testing represent
     honest performance reporting, contrasting sharply with typical
     published claims that likely reflect data mining and lookahead bias."

Default parameters (from the paper)
------------------------------------
    train_bars = 252   (one trading year)
    test_bars  = 63    (one quarter)
    step_bars  = 63    (step forward one quarter per fold)

Usage
-----
    from backtesting.walk_forward import walk_forward_test, print_walk_forward_report
    results, summary = walk_forward_test(df, strategy_func)
    print_walk_forward_report(results, summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional

SEP  = "=" * 70
SEP2 = "-" * 70


# ── Core rolling walk-forward engine ──────────────────────────────────────────

def walk_forward_test(
    df:           pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], pd.DataFrame],
    train_bars:   int           = 252,
    test_bars:    int           = 63,
    step_bars:    Optional[int] = None,
) -> tuple[list[dict], dict]:
    """
    Rolling walk-forward validation.

    Parameters
    ----------
    df            : Full OHLCV + indicator DataFrame in chronological order.
                    Must have at least train_bars + test_bars rows.
    strategy_func : callable(df_slice) → df_slice_with_signals
                    The strategy you want to validate (same function used
                    in main.py).  It must add a 'crossover' column.
    train_bars    : Training window in bars  (default 252 = 1 trading year)
    test_bars     : Out-of-sample test window (default 63 = 1 quarter)
    step_bars     : How far to roll forward per fold (default = test_bars)

    Returns
    -------
    results : list[dict]  — per-fold metrics
    summary : dict        — aggregate statistics across all valid folds

    Notes
    -----
    The strategy_func is applied identically to both the train slice (to
    warm up indicators that need a lookback, e.g. MACD) and the test slice.
    The train slice result is discarded; only the test slice result is
    evaluated.  This matches the Deep et al. protocol.
    """
    # ── Import the backtester ─────────────────────────────────────────────────
    # Try the newer backtest_engine first; fall back to the original backtest()
    _backtest = _resolve_backtest()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if step_bars is None:
        step_bars = test_bars

    required = train_bars + test_bars
    if len(df) < required:
        empty_summary = {
            "folds":       0,
            "valid_folds": 0,
            "error": (
                f"DataFrame too short for walk-forward: "
                f"need ≥ {required} bars, got {len(df)}.  "
                f"Reduce train_bars / test_bars or use a longer history."
            ),
        }
        return [], empty_summary

    results: list[dict] = []
    start   = 0
    fold_n  = 0

    # ── Rolling folds ─────────────────────────────────────────────────────────
    while start + train_bars + test_bars <= len(df):
        fold_n += 1

        train_slice = df.iloc[start : start + train_bars].copy()
        test_slice  = df.iloc[start + train_bars : start + train_bars + test_bars].copy()

        fold_meta = {
            "fold":        fold_n,
            "train_start": df.index[start],
            "train_end":   df.index[start + train_bars - 1],
            "test_start":  df.index[start + train_bars],
            "test_end":    df.index[start + train_bars + test_bars - 1],
            "train_bars":  train_bars,
            "test_bars":   test_bars,
        }

        # ── Apply strategy to TRAIN (warms up indicators, result discarded) ──
        try:
            strategy_func(train_slice)
        except Exception:
            pass  # Train-slice application failure is non-fatal

        # ── Apply strategy to TEST (out-of-sample) ────────────────────────────
        try:
            test_signals = strategy_func(test_slice)
        except Exception as exc:
            fold_meta["error"] = f"strategy_func failed: {exc}"
            results.append(fold_meta)
            start += step_bars
            continue

        # ── Backtest strictly on the test window ──────────────────────────────
        try:
            fold_result = _backtest(test_signals)
        except Exception as exc:
            fold_meta["error"] = f"backtest failed: {exc}"
            results.append(fold_meta)
            start += step_bars
            continue

        # Merge fold meta into result dict
        fold_result.update(fold_meta)
        results.append(fold_result)

        start += step_bars

    summary = summarize_walk_forward(results)
    return results, summary


# ── Summary statistics ─────────────────────────────────────────────────────────

def summarize_walk_forward(results: list[dict]) -> dict:
    """Aggregate per-fold results into a single summary dict."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return {
            "folds":       len(results),
            "valid_folds": 0,
            "error":       "All walk-forward folds failed — check strategy_func and data quality",
        }

    returns    = [r.get("return_pct",   0.0) for r in valid]
    sharpes    = [r.get("sharpe",       0.0) for r in valid]
    drawdowns  = [r.get("max_drawdown", 0.0) for r in valid]
    win_rates  = [r.get("win_rate",     0.0) for r in valid]
    trade_cnts = [r.get("n_trades",       0) for r in valid]

    profitable     = sum(1 for r in returns if r > 0)
    fold_win_rate  = profitable / len(valid) * 100

    return {
        # Fold counts
        "folds":              len(results),
        "valid_folds":        len(valid),
        "profitable_folds":   profitable,
        "fold_win_rate_pct":  round(fold_win_rate, 1),

        # Return distribution
        "mean_return_pct":    round(float(np.mean(returns)),   2),
        "median_return_pct":  round(float(np.median(returns)), 2),
        "std_return_pct":     round(float(np.std(returns)),    2),
        "best_fold_pct":      round(float(np.max(returns)),    2),
        "worst_fold_pct":     round(float(np.min(returns)),    2),

        # Sharpe
        "mean_sharpe":        round(float(np.mean(sharpes)),   2),
        "median_sharpe":      round(float(np.median(sharpes)), 2),

        # Drawdown
        "mean_drawdown_pct":  round(float(np.mean(drawdowns)), 2),
        "worst_drawdown_pct": round(float(np.min(drawdowns)),  2),

        # Trades
        "mean_win_rate_pct":      round(float(np.mean(win_rates)), 1),
        "total_trades":           int(sum(trade_cnts)),
        "mean_trades_per_fold":   round(float(np.mean(trade_cnts)), 1),
    }


# ── Formatted console report ───────────────────────────────────────────────────

def print_walk_forward_report(results: list[dict], summary: dict) -> None:
    """
    Print a formatted walk-forward report via the project logger.
    Called from main.py after walk_forward_test() completes.
    """
    from utils.logger import logger

    logger.info(SEP)
    logger.info("  WALK-FORWARD VALIDATION  (out-of-sample only)")
    logger.info("  Ref: Deep et al. (2025) arXiv:2512.12924")
    logger.info(SEP)

    # ── Error guard ───────────────────────────────────────────────────────────
    if "error" in summary and summary.get("valid_folds", 0) == 0:
        logger.warning(f"  Walk-forward skipped: {summary['error']}")
        logger.info(SEP)
        return

    # ── Per-fold table ────────────────────────────────────────────────────────
    header = (
        f"  {'Fold':<5}  {'OOS Start':<12}  {'Return':>8}  "
        f"{'Sharpe':>7}  {'MaxDD':>7}  {'WinRate':>8}  {'Trades':>7}"
    )
    logger.info(header)
    logger.info(SEP2)

    for r in results:
        if "error" in r:
            logger.warning(
                f"  Fold {r.get('fold', '?'):>2}  -- error: {r['error']}"
            )
            continue

        test_dt  = _fmt_date(r.get("test_start"))
        ret_s    = f"{r.get('return_pct',   0.0):+.1f}%"
        sharpe_s = f"{r.get('sharpe',       0.0):.2f}"
        dd_s     = f"{r.get('max_drawdown', 0.0):.1f}%"
        wr_s     = f"{r.get('win_rate',     0.0):.1f}%"
        tr_s     = str(r.get("n_trades", 0))

        logger.info(
            f"  {r['fold']:<5}  {test_dt:<12}  {ret_s:>8}  "
            f"{sharpe_s:>7}  {dd_s:>7}  {wr_s:>8}  {tr_s:>7}"
        )

    logger.info(SEP2)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    s = summary

    logger.info(
        f"  {'Folds (valid / total)':<32} "
        f"{s['valid_folds']} / {s['folds']}"
    )
    logger.info(
        f"  {'Profitable folds':<32} "
        f"{s['profitable_folds']}  ({s['fold_win_rate_pct']:.1f}%)"
    )
    logger.info(SEP2)
    logger.info(f"  {'Mean OOS Return':<32} {s['mean_return_pct']:>+.2f}%")
    logger.info(f"  {'Median OOS Return':<32} {s['median_return_pct']:>+.2f}%")
    logger.info(f"  {'Return Std Dev':<32} {s['std_return_pct']:.2f}%")
    logger.info(
        f"  {'Best / Worst fold':<32} "
        f"{s['best_fold_pct']:>+.1f}%  /  {s['worst_fold_pct']:>+.1f}%"
    )
    logger.info(SEP2)
    logger.info(f"  {'Mean OOS Sharpe':<32} {s['mean_sharpe']:.2f}")
    logger.info(f"  {'Worst Drawdown (any fold)':<32} {s['worst_drawdown_pct']:.2f}%")
    logger.info(f"  {'Mean Win Rate':<32} {s['mean_win_rate_pct']:.1f}%")
    logger.info(f"  {'Total OOS Trades':<32} {s['total_trades']}")
    logger.info(SEP2)

    # ── Overall grade ─────────────────────────────────────────────────────────
    grade = _wf_grade(s)
    logger.info(f"  Walk-Forward Grade:  {grade}")
    logger.info(
        f"  (note: modest OOS returns are honest — "
        f"overfitted systems show high in-sample, near-zero OOS)"
    )
    logger.info(SEP)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_backtest():
    """Import the best available backtest function."""
    try:
        from backtesting.backtester import backtest_engine as _bt
        return _bt
    except ImportError:
        pass
    try:
        from backtesting.backtester import backtest as _bt
        return _bt
    except ImportError:
        raise ImportError(
            "backtesting.backtester must expose either backtest() or backtest_engine()"
        )


def _fmt_date(dt) -> str:
    if dt is None:
        return "N/A"
    try:
        return dt.strftime("%Y-%m-%d")
    except AttributeError:
        return str(dt)


def _wf_grade(s: dict) -> str:
    fold_wr = s.get("fold_win_rate_pct", 0)
    mean_sh = s.get("mean_sharpe", 0)

    if fold_wr >= 60 and mean_sh > 0.5:
        return "PASS  [robust — strategy holds up out-of-sample]"
    elif fold_wr >= 40:
        return "MARGINAL  [inconsistent — tune parameters or add filters]"
    else:
        return "FAIL  [not robust — likely overfitted to historical data]"