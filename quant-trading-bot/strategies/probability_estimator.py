"""
Probability Estimator — directional signal quality gate.
=========================================================
Drop-in replacement for the original Brownian-motion estimator.

The original estimator used a Gaussian CDF on the indicator gap
(gap_now, drift, vol).  It produced 0% outputs because:
    1. gap_now can be very large or very small → z becomes extreme → CDF ≈ 0 or 1
    2. vol ≤ 0 edge-cases returned 0% with no warning
    3. The regime-aware assignment was inverted for some MACD configurations

This module replaces the formula with a LogisticProbabilityModel
(scikit-learn) trained on 6 engineered features.  When scikit-learn is
not installed it falls back to a historical base-rate estimate so the
system degrades gracefully rather than crashing.

Interface preserved
-------------------
The function ``estimate_crossover_probability(df)`` returns exactly the
same dict shape as the original:

    {
        "buy_3d_pct":  float  (0 – 100)
        "sell_3d_pct": float  (0 – 100)
        "explanation": str
    }

All callers in main.py and risk_manager.py continue to work unchanged.

Usage
-----
    from strategies.probability_estimator import estimate_crossover_probability
    probs = estimate_crossover_probability(df)
    # probs["buy_3d_pct"]  → 63.2   (strong bullish signal)
    # probs["sell_3d_pct"] → 36.8
    # probs["explanation"] → "LogisticModel [trained] | P(up)=63.2% ..."

Reset per ticker (multi-ticker mode)
-------------------------------------
    from strategies.probability_estimator import reset_probability_model
    reset_probability_model()
"""

from __future__ import annotations

from config.settings import (
    PROB_TRAIN_BARS,
    PROB_BUY_THRESHOLD,
    PROB_SELL_THRESHOLD,
)
from models.logistic_probability import get_or_train_model, reset_model


# ── Public API ─────────────────────────────────────────────────────────────────

def estimate_crossover_probability(df) -> dict:
    """
    Estimate directional probability using logistic regression.

    The model is trained once (lazily, on first call) on the last
    PROB_TRAIN_BARS bars of df, then cached for the remainder of the run.
    Subsequent calls use the cached model — only the last row of df is
    evaluated for prediction.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicators DataFrame.  Required columns:  close
        Optional (used as features if present):
            short_ma, long_ma, rsi, volume, high, low

    Returns
    -------
    dict
        buy_3d_pct  : float 0–100  estimated P(up move)
        sell_3d_pct : float 0–100  estimated P(down move)
        explanation : str          human-readable one-liner for the log
    """
    if len(df) < 30:
        return {
            "buy_3d_pct":  0.0,
            "sell_3d_pct": 0.0,
            "explanation": "Insufficient history (need ≥ 30 bars)",
        }

    # Thresholds stored in settings as integers (55, 45); convert to fractions
    buy_thresh  = PROB_BUY_THRESHOLD  / 100.0
    sell_thresh = PROB_SELL_THRESHOLD / 100.0

    model   = get_or_train_model(df, PROB_TRAIN_BARS, buy_thresh, sell_thresh)
    up_prob = model.predict(df)       # P(next bar closes UP)
    dn_prob = 1.0 - up_prob

    buy_pct  = round(up_prob * 100, 1)
    sell_pct = round(dn_prob * 100, 1)

    # ── Determine current crossover context ───────────────────────────────────
    direction = "no crossover"
    if "crossover" in df.columns:
        last_cross = int(df["crossover"].iloc[-1])
        if last_cross == 1:
            direction = "bullish crossover"
        elif last_cross == -1:
            direction = "bearish crossover"

    # ── Confidence label ──────────────────────────────────────────────────────
    edge = abs(up_prob - 0.5)
    confidence = (
        "high confidence"     if edge > 0.15 else
        "moderate confidence" if edge > 0.07 else
        "low confidence (near 50/50)"
    )

    # ── Model status tag ──────────────────────────────────────────────────────
    if model.is_trained:
        status = f"trained on {PROB_TRAIN_BARS} bars"
    else:
        status = f"fallback — {model.last_error}"

    explanation = (
        f"LogisticModel [{status}]  |  "
        f"P(up)={buy_pct}%  P(dn)={sell_pct}%  |  "
        f"Signal: {direction}  |  {confidence}"
    )

    return {
        "buy_3d_pct":  buy_pct,
        "sell_3d_pct": sell_pct,
        "explanation": explanation,
    }


def reset_probability_model() -> None:
    """
    Clear the cached model.

    Call this before processing a new ticker in multi-ticker mode so
    each ticker gets its own model trained on its own price history.

    Example:
        for ticker, df in ticker_data.items():
            reset_probability_model()
            probs = estimate_crossover_probability(df)
    """
    reset_model()