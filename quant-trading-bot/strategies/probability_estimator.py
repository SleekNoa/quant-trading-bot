"""
Improved 3-day crossover probability estimator.
Uses Brownian motion with drift on the indicator gap.
Logic now correctly distinguishes current regime and target direction.
"""

import numpy as np
from scipy.special import erf
from config.settings import STRATEGY

def _norm_cdf(x: float) -> float:
    """Pure NumPy approximation of normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def estimate_crossover_probability(df):
    """
    Returns dict with:
        buy_3d_pct:   float (0–100) — estimated prob of bullish crossover in next 3 days
        sell_3d_pct:  float (0–100) — estimated prob of bearish crossover in next 3 days
        explanation:  str
    """
    if len(df) < 20:
        return {
            "buy_3d_pct": 0,
            "sell_3d_pct": 0,
            "explanation": "Not enough history (need ≥20 bars for drift/vol)"
        }

    latest = df.iloc[-1]
    recent = df.iloc[-20:]   # short window for local drift/vol

    gap_series = None
    strategy_name = "Unknown"

    if STRATEGY == "macd" and "macd" in latest and "macd_signal" in latest:
        gap_series = recent["macd"] - recent["macd_signal"]
        strategy_name = "MACD"
    elif STRATEGY in ("sma", "moving_average") and all(c in latest for c in ["short_ma", "long_ma"]):
        gap_series = recent["short_ma"] - recent["long_ma"]
        strategy_name = "SMA"
    # Add other strategies here when they have a meaningful signed gap

    if gap_series is None or gap_series.isna().all():
        # Fallback — very rough historical base rate
        sig_count = (df["crossover"] != 0).sum()
        if sig_count < 5:
            return {"buy_3d_pct": 0, "sell_3d_pct": 0, "explanation": "Too few signals for fallback rate"}
        avg_bars_per_signal = len(df) / sig_count
        base_prob = min(1.0, 3.0 / avg_bars_per_signal)
        return {
            "buy_3d_pct": round(base_prob * 50, 1),
            "sell_3d_pct": round(base_prob * 50, 1),
            "explanation": f"Historical base rate (~1 signal every {avg_bars_per_signal:.1f} bars)"
        }

    gap_now = gap_series.iloc[-1]
    drift = gap_series.diff().mean()           # daily change in gap
    vol = gap_series.diff().std()              # volatility of gap changes

    if vol <= 0 or np.isnan(vol):
        return {"buy_3d_pct": 0, "sell_3d_pct": 0, "explanation": "No volatility in gap"}

    T = 3.0
    # Probability that gap reaches / crosses zero in time T
    # We use the sign-consistent formulation
    z = (-gap_now - drift * T) / (vol * np.sqrt(T))
    prob_cross = _norm_cdf(z)   # P(gap ≤ 0 at time T | current gap_now)

    # ── Regime-aware assignment ───────────────────────────────────────────────
    display_prob = round(min(prob_cross * 100, 70.0), 1)  # single capped value used everywhere

    if gap_now > 0:
        # Already bullish (positive gap) → crossing zero means bearish crossover
        buy_3d_pct  = 0.0
        sell_3d_pct = display_prob
        regime      = "Bullish"
        forecast    = f"Bearish crossover in 3 days: {display_prob}%"
    elif gap_now < 0:
        # Already bearish → crossing zero means bullish crossover
        buy_3d_pct  = display_prob
        sell_3d_pct = 0.0
        regime      = "Bearish"
        forecast    = f"Bullish crossover in 3 days: {display_prob}%"
    else:
        # Exactly at zero — symmetric
        buy_3d_pct  = display_prob
        sell_3d_pct = display_prob
        regime      = "Neutral"
        forecast    = f"Crossover either direction in 3 days: {display_prob}%"

    # ── Strength qualifier (based on capped display_prob for consistency) ─────
    strength = ""
    if display_prob >= 65:
        strength = " (strong reversal pressure)"
    elif display_prob >= 50:
        strength = " (moderate reversal pressure)"
    elif display_prob >= 35:
        strength = " (mild reversal pressure)"

    # ── Human-readable explanation ────────────────────────────────────────────
    expl = (
        f"{strategy_name} gap = {gap_now:.3f}  |  "
        f"z = {z:.2f}  |  μ = {drift:.4f}  |  σ = {vol:.4f}  |  "
        f"Current regime: {regime}  →  {forecast}{strength}"
    )

    return {
        "buy_3d_pct": buy_3d_pct,
        "sell_3d_pct": sell_3d_pct,
        "explanation": expl
    }