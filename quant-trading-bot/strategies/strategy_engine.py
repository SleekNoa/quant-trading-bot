"""
strategies/strategy_engine.py — Multi-Strategy Plugin Engine
"""
import time
import pandas as pd
from utils.logger import logger

_REGISTRY = {}
_SCORES   = {"BUY": 1, "SELL": -1, "HOLD": 0}

_REGIME_MULTIPLIERS = {
    "trending": {
        "MACD": 1.5, "SMA": 1.3, "DirectionalChange": 2.0,
        "RSI": 0.7,  "Bollinger": 0.7, "Stochastic": 0.8,
    },
    "ranging": {
        "MACD": 0.7, "SMA": 0.6, "DirectionalChange": 0.8,
        "RSI": 1.5,  "Bollinger": 1.5, "Stochastic": 1.3,
    },
    "unknown": {},
}

BUY_THRESHOLD  =  1.5
SELL_THRESHOLD = -1.5


def register_strategy(name, weight=1.0, enabled=True):
    def decorator(fn):
        _REGISTRY[name] = {"fn": fn, "weight": weight, "enabled": enabled}
        return fn
    return decorator


def set_strategy_enabled(name, enabled):
    if name in _REGISTRY:
        _REGISTRY[name]["enabled"] = enabled


def list_strategies():
    return list(_REGISTRY.keys())


def evaluate_strategies(df, regime="unknown"):
    multipliers    = _REGIME_MULTIPLIERS.get(regime, {})
    signals        = {}
    weighted_score = 0.0
    total_weight   = 0.0

    for name, entry in _REGISTRY.items():
        if not entry["enabled"]:
            signals[name] = {"signal": "DISABLED", "base_weight": 0, "regime_mult": 0,
                             "eff_weight": 0, "contribution": 0, "elapsed_ms": 0}
            continue

        t0 = time.perf_counter()
        try:
            raw_signal = entry["fn"](df)
            if raw_signal not in ("BUY", "SELL", "HOLD"):
                raw_signal = "HOLD"
        except Exception as e:
            logger.warning(f"[engine] Strategy '{name}' raised: {e}")
            raw_signal = "HOLD"
        elapsed_ms = (time.perf_counter() - t0) * 1000

        base_w   = entry["weight"]
        regime_m = multipliers.get(name, 1.0)
        eff_w    = base_w * regime_m
        contrib  = _SCORES[raw_signal] * eff_w

        weighted_score += contrib
        total_weight   += eff_w

        signals[name] = {
            "signal":       raw_signal,
            "base_weight":  round(base_w, 2),
            "regime_mult":  round(regime_m, 2),
            "eff_weight":   round(eff_w, 2),
            "contribution": round(contrib, 3),
            "elapsed_ms":   round(elapsed_ms, 1),
        }

    if weighted_score >= BUY_THRESHOLD:
        decision = "BUY"
    elif weighted_score <= SELL_THRESHOLD:
        decision = "SELL"
    else:
        decision = "HOLD"

    report = {
        "decision":       decision,
        "weighted_score": round(weighted_score, 3),
        "total_weight":   round(total_weight, 2),
        "regime":         regime,
        "buy_votes":      sum(1 for s in signals.values() if s["signal"] == "BUY"),
        "sell_votes":     sum(1 for s in signals.values() if s["signal"] == "SELL"),
        "hold_votes":     sum(1 for s in signals.values() if s["signal"] == "HOLD"),
        "strategies":     signals,
        "thresholds":     {"buy": BUY_THRESHOLD, "sell": SELL_THRESHOLD},
    }
    return decision, report


def log_engine_report(report):
    SEP = "-" * 70
    d   = report["decision"]
    logger.info(SEP)
    logger.info(f"  STRATEGY ENGINE  =>  {d}  "
                f"(score:{report['weighted_score']:+.2f}  regime:{report['regime']})")
    logger.info(SEP)
    logger.info(f"  {'Strategy':<22} {'Signal':<8} {'Wt':>4}  {'xReg':>5}  {'Contrib':>8}  {'ms':>5}")
    logger.info(f"  {'─'*22} {'─'*8} {'─'*4}  {'─'*5}  {'─'*8}  {'─'*5}")
    for name, s in report["strategies"].items():
        if s["signal"] == "DISABLED":
            continue
        logger.info(
            f"  {name:<22} {s['signal']:<8} {s['base_weight']:>4.1f}  "
            f"{s['regime_mult']:>5.1f}  {s['contribution']:>+8.3f}  {s['elapsed_ms']:>5.1f}"
        )
    logger.info(SEP)
    logger.info(
        f"  BUY:{report['buy_votes']}  SELL:{report['sell_votes']}  "
        f"HOLD:{report['hold_votes']}  |  total:{report['weighted_score']:+.2f}  "
        f"(need >{report['thresholds']['buy']})"
    )
    logger.info(SEP)


def detect_market_regime(df, adx_threshold=25.0):
    if "adx" not in df.columns:
        return "unknown"
    latest_adx = df["adx"].iloc[-1]
    if pd.isna(latest_adx):
        return "unknown"
    regime = "trending" if latest_adx >= adx_threshold else "ranging"
    return regime