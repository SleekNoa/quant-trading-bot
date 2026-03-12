"""
risk/risk_manager.py — Institutional Risk Management Framework
==============================================================
Implements the quantitative risk framework from:

    Wang, Zhao & Wang (2026).
    "Integrated financial risk management framework: A quantitative
    portfolio optimization analysis."
    Case Studies in Thermal Engineering. ScienceDirect.

Key methods implemented:
    • Historical Value-at-Risk (VaR) at configurable confidence level
    • Conditional Value-at-Risk (CVaR / Expected Shortfall) for tail risk
    • VaR-based position sizing (replaces flat % risk)
    • Portfolio-level drawdown circuit-breaker
    • Daily loss circuit-breaker
    • Monte Carlo stress test (pre-trade gate)
    • Signal quality gate (probability threshold)

The paper reports: Sharpe 1.85, max drawdown reduction 18.7%,
CVaR improvement 24.3% through these methods vs benchmark.
"""

import numpy as np
import pandas as pd
from utils.logger import logger
from config.settings import (
    VAR_CONFIDENCE,
    VAR_LOOKBACK_DAYS,
    CVAR_TAIL_PCT,
    MAX_RISK_PER_TRADE_PCT,
    MAX_POSITION_PCT,
    MAX_PORTFOLIO_DRAWDOWN_PCT,
    DAILY_LOSS_LIMIT_PCT,
    MIN_SIGNAL_PROBABILITY,
    STOP_LOSS_PCT,
    RUN_STRESS_TEST,
    MC_SIMULATIONS,
    MC_HORIZON_DAYS,
    MC_ACCEPTABLE_VAR_PCT,
    INITIAL_CAPITAL,
)


# ══════════════════════════════════════════════════════════════════════
#  1. VaR & CVaR
# ══════════════════════════════════════════════════════════════════════

def compute_historical_var(df: pd.DataFrame, confidence: float = VAR_CONFIDENCE,
                           lookback: int = VAR_LOOKBACK_DAYS) -> float:
    """
    Historical Value-at-Risk using rolling daily returns.

    Returns the 1-day VaR as a positive decimal (e.g. 0.023 = 2.3% loss).
    Wang et al. use historical simulation as the base-case VaR estimator
    before applying copula-based dependency modeling for portfolio VaR.
    """
    if "close" not in df.columns or len(df) < lookback + 1:
        logger.warning("[risk] Not enough data for VaR — using fallback 3%")
        return 0.03

    returns = df["close"].pct_change().dropna().tail(lookback)
    # VaR = loss at the (1 - confidence) percentile of the return distribution
    var = float(-np.percentile(returns, (1 - confidence) * 100))
    return max(var, 0.001)   # floor at 0.1% to avoid divide-by-zero


def compute_cvar(df: pd.DataFrame, confidence: float = VAR_CONFIDENCE,
                 lookback: int = VAR_LOOKBACK_DAYS) -> float:
    """
    Conditional Value-at-Risk (Expected Shortfall).

    CVaR = mean of losses that exceed the VaR threshold.
    Wang et al. report a 24.3% improvement in CVaR through derivative
    overlay strategies; we use CVaR here to set a tighter stop floor.

    Returns positive decimal (e.g. 0.041 = 4.1% expected tail loss).
    """
    if "close" not in df.columns or len(df) < lookback + 1:
        return 0.05

    returns = df["close"].pct_change().dropna().tail(lookback)
    var_threshold = -compute_historical_var(df, confidence, lookback)
    tail_losses   = returns[returns <= var_threshold]

    if len(tail_losses) == 0:
        return compute_historical_var(df, confidence, lookback) * 1.5

    cvar = float(-tail_losses.mean())
    return max(cvar, 0.001)


# ══════════════════════════════════════════════════════════════════════
#  2. Position Sizing
# ══════════════════════════════════════════════════════════════════════

def calculate_position_size(balance: float, price: float,
                             df: pd.DataFrame = None) -> int:
    """
    VaR-based position sizing.

    Replaces the old flat-percent approach with a method derived from
    Wang et al.'s risk-return optimization framework:

        shares = (portfolio × MAX_RISK_PER_TRADE_PCT) / (price × daily_VaR)

    This ensures 1-day losses on the position stay within the risk budget
    at the configured confidence level, regardless of asset volatility.

    Falls back to simple % sizing if no price history is available.
    """
    if df is not None and len(df) > VAR_LOOKBACK_DAYS:
        daily_var = compute_historical_var(df)
        # dollar risk budget ÷ expected dollar loss per share
        risk_budget  = balance * MAX_RISK_PER_TRADE_PCT
        loss_per_share = price * daily_var
        if loss_per_share > 0:
            shares = int(risk_budget / loss_per_share)
        else:
            shares = 0
    else:
        # Fallback: simple % of balance (old behaviour)
        shares = int((balance * MAX_RISK_PER_TRADE_PCT) / price)

    # Hard cap: never exceed MAX_POSITION_PCT of portfolio in one trade
    max_shares = int((balance * MAX_POSITION_PCT) / price)
    shares = min(shares, max_shares)

    if shares > 0:
        dollar_exposure = shares * price
        pct_of_portfolio = dollar_exposure / balance * 100
        logger.info(
            f"[risk] VaR sizing → {shares} shares  "
            f"(${dollar_exposure:,.0f}  =  {pct_of_portfolio:.1f}% of portfolio)"
        )

    return max(shares, 0)


def full_position_size(balance: float, price: float) -> int:
    """Buy as many shares as balance allows (backtester only)."""
    return int(balance / price)


# ══════════════════════════════════════════════════════════════════════
#  3. Circuit Breakers
# ══════════════════════════════════════════════════════════════════════

def check_drawdown_circuit_breaker(account_equity: float,
                                    peak_equity: float = None) -> bool:
    """
    Portfolio-level drawdown circuit-breaker.

    Wang et al. implement automated circuit-breakers that halt trading
    when portfolio drawdown exceeds a configured threshold. This prevents
    compounding losses in adverse market regimes.

    Returns True if trading should be HALTED (drawdown limit breached).
    Returns False if trading is ALLOWED.
    """
    if peak_equity is None:
        peak_equity = INITIAL_CAPITAL

    if peak_equity <= 0:
        return False

    drawdown = (account_equity - peak_equity) / peak_equity
    if drawdown <= -MAX_PORTFOLIO_DRAWDOWN_PCT:
        logger.warning(
            f"[risk] ⛔ CIRCUIT BREAKER — portfolio drawdown "
            f"{drawdown*100:.1f}% exceeds limit "
            f"-{MAX_PORTFOLIO_DRAWDOWN_PCT*100:.0f}%. Trading HALTED."
        )
        return True
    return False


def check_daily_loss_limit(daily_pnl: float, account_equity: float) -> bool:
    """
    Daily loss circuit-breaker.

    Halts trading for the rest of the session if intraday P&L
    falls below DAILY_LOSS_LIMIT_PCT of portfolio value.

    Returns True if trading should HALT for the day.
    """
    if account_equity <= 0:
        return False

    daily_pnl_pct = daily_pnl / account_equity
    if daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
        logger.warning(
            f"[risk] ⛔ DAILY LOSS LIMIT — P&L {daily_pnl_pct*100:.2f}% "
            f"exceeds -{DAILY_LOSS_LIMIT_PCT*100:.0f}%. No more trades today."
        )
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
#  4. Signal Quality Gate
# ══════════════════════════════════════════════════════════════════════

def passes_signal_gate(buy_pct: float, sell_pct: float,
                        crossover: int) -> bool:
    """
    Reject low-conviction signals based on crossover probability.

    The probability estimator gives us a forward-looking confidence score.
    We only execute trades where conviction exceeds MIN_SIGNAL_PROBABILITY.
    This directly implements the "dynamic signal validation" concept from
    Wang et al.'s integrated framework.

    Returns True if the signal is strong enough to trade.
    """
    if MIN_SIGNAL_PROBABILITY <= 0:
        return True   # gate disabled

    if crossover == 1 and buy_pct >= MIN_SIGNAL_PROBABILITY:
        logger.info(f"[risk] ✅ Signal gate passed — buy probability {buy_pct:.1f}%")
        return True
    elif crossover == -1 and sell_pct >= MIN_SIGNAL_PROBABILITY:
        logger.info(f"[risk] ✅ Signal gate passed — sell probability {sell_pct:.1f}%")
        return True
    else:
        relevant_pct = buy_pct if crossover == 1 else sell_pct
        logger.warning(
            f"[risk] ❌ Signal gate REJECTED — probability {relevant_pct:.1f}% "
            f"< minimum {MIN_SIGNAL_PROBABILITY:.0f}%"
        )
        return False


# ══════════════════════════════════════════════════════════════════════
#  5. Monte Carlo Stress Test (Pre-Trade Gate)
# ══════════════════════════════════════════════════════════════════════

def monte_carlo_stress_test(df: pd.DataFrame, price: float,
                             shares: int) -> dict:
    """
    Lightweight Monte Carlo stress test before committing to a trade.

    Simulates MC_SIMULATIONS price paths over MC_HORIZON_DAYS using
    historical drift and volatility from recent price action.

    Inspired by Wang et al.'s Monte Carlo + copula simulation framework
    for assessing portfolio performance under macroeconomic scenarios.

    Returns dict:
        passed      – bool (True = proceed with trade)
        var_pct     – simulated VaR over horizon as % of position
        worst_pct   – worst simulated outcome (%)
        median_pct  – median simulated outcome (%)
        explanation – human-readable summary
    """
    if not RUN_STRESS_TEST or df is None or len(df) < 30:
        return {"passed": True, "var_pct": 0, "worst_pct": 0,
                "median_pct": 0, "explanation": "Stress test skipped"}

    returns = df["close"].pct_change().dropna().tail(60)
    mu      = float(returns.mean())
    sigma   = float(returns.std())

    np.random.seed(None)
    daily_shocks = np.random.normal(mu, sigma, (MC_SIMULATIONS, MC_HORIZON_DAYS))
    # Simulate cumulative price paths
    cumulative   = np.prod(1 + daily_shocks, axis=1)   # shape: (MC_SIMULATIONS,)
    path_returns = cumulative - 1.0

    var_pct    = float(-np.percentile(path_returns, (1 - VAR_CONFIDENCE) * 100))
    worst_pct  = float(path_returns.min())
    median_pct = float(np.median(path_returns))

    position_value = shares * price
    passed = var_pct < MC_ACCEPTABLE_VAR_PCT

    status = "✅ PASSED" if passed else "❌ BLOCKED"
    explanation = (
        f"MC stress test ({MC_SIMULATIONS} paths, {MC_HORIZON_DAYS}d horizon): "
        f"VaR={var_pct*100:.1f}%  worst={worst_pct*100:.1f}%  "
        f"median={median_pct*100:.1f}%  "
        f"position=${position_value:,.0f}  →  {status}"
    )
    logger.info(f"[risk] {explanation}")

    if not passed:
        logger.warning(
            f"[risk] ⛔ MC VaR {var_pct*100:.1f}% > limit "
            f"{MC_ACCEPTABLE_VAR_PCT*100:.0f}% — trade BLOCKED by stress test"
        )

    return {
        "passed":      passed,
        "var_pct":     var_pct,
        "worst_pct":   worst_pct,
        "median_pct":  median_pct,
        "explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════════════
#  6. Stop-Loss Price Calculator
# ══════════════════════════════════════════════════════════════════════

def get_stop_loss_price(entry_price: float) -> float:
    """Return the hard stop-loss price level for a given entry."""
    return round(entry_price * (1 - STOP_LOSS_PCT), 4)