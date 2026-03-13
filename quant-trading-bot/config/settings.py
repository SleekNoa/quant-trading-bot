"""
QuantBot — Central Configuration
==================================
All tunable parameters live here.  main.py imports from this file only;
individual modules also import specific constants they need.

Environment variables (.env) override any defaults below when set.

Sections
--------
    1.  API Credentials
    2.  Symbol / Universe
    3.  Strategy Selection & Parameters
    4.  ADX / OBV Filters
    5.  Data Source
    6.  Capital & Risk
    7.  VaR / CVaR Position Sizing          ← risk_manager.py
    8.  Logistic Probability Model          ← probability_estimator.py (NEW)
    9.  Walk-Forward Validation             ← backtesting/walk_forward.py (NEW)
    10. Monte Carlo Robustness Test         ← backtesting/monte_carlo.py  (NEW)
    11. Multi-Ticker Engine                 ← data/multi_ticker.py        (NEW)
    12. Circuit-Breakers & Stop-Loss
    13. Live Execution Flags
"""

import os
from dotenv import load_dotenv
import pandas as pd
from utils import load_tickers






load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# 1.  API Credentials
# ══════════════════════════════════════════════════════════════════════════════

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
ALPACA_API_KEY       = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY    = os.getenv("ALPACA_SECRET_KEY")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Symbol / Universe
# ══════════════════════════════════════════════════════════════════════════════

# Primary symbol for the single-ticker pipeline (reads from .env if set)
SYMBOL = os.getenv("SYMBOL", "AAPL")

# Multi-ticker universe — used when USE_MULTI_TICKER = True
# The bot scans all tickers, ranks signals, then executes on the best one
#TICKERS = [
 # "ACHR","LIF","IONQ","MVIS","BBBYQ","HILS","SNTI","GFAI","CXAI","AIKI",
#"FFIE","EVGO","MULN","HCDI","KAVL","CRKN","SOPA","VRAX","IMPP","CEI",
#"SIDU","ATNF","RETO","VINE","HUSA"

#]

TICKERS = load_tickers()

# holding {DO NOT REMOVE} - LIN, XOM
# GREAT TEST {DO NOT REMOVE} - ["LIF", "DOW", "LYB", "CE", "OLN", "ACHR", "APD", "DD", "HUN",
#            "WLK", "EMN", "ALB", "SQM", "TROX", "CBT", "EPD", "PPG", "SHW", "CC", "KRO"]

# Capital allocation method when multiple signals are active
# Options: "equal" | "score_weighted" | "sharpe_weighted"
TICKER_ALLOC_METHOD = "equal"

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Strategy Selection & Parameters
# ══════════════════════════════════════════════════════════════════════════════

ASK_SAME_DAY_CONFIRM = True # Set to False to disable the [Y/N] prompt entirely

# Change STRATEGY to switch the active strategy without touching main.py
# Options: "sma" | "rsi" | "macd" | "bollinger" | "stochastic"
STRATEGY = "macd"

# SMA parameters
SHORT_WINDOW = 20
LONG_WINDOW  = 50

# RSI parameters
RSI_PERIOD     = 14
RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70

# MACD parameters
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Bollinger Bands parameters
BB_PERIOD = 20
BB_STD    = 2.0

# Stochastic parameters
STOCH_K = 14
STOCH_D = 3

# ══════════════════════════════════════════════════════════════════════════════
# 4.  ADX / OBV Filters
# ══════════════════════════════════════════════════════════════════════════════

# ADX measures trend STRENGTH (not direction).
# Signals are suppressed when ADX < ADX_THRESHOLD (choppy/ranging market).
ADX_PERIOD    = 14
ADX_THRESHOLD = 28     # below 25 = choppy → momentum indicators produce false signals

# OBV rising = volume confirms the price move.
OBV_MA_PERIOD = 5      # smoothing window for OBV trend detection

# Filter enable/disable flags
USE_ADX_FILTER = True
USE_OBV_FILTER = True

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Data Source
# ══════════════════════════════════════════════════════════════════════════════

# True  = always use 500-bar synthetic random walk  (offline testing)
# False = live data via yfinance / Alpha Vantage     (real OHLCV)
USE_SIMULATED_DATA = False

# Legacy flag used by some modules — kept for backward compatibility
SHOW_ZSCORE_IN_PROB = True

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Capital & Risk
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL  = 100_000   # starting paper-trade capital ($)
RISK_PERCENT     = 0.01     # legacy flat-risk fallback (1% of balance per trade)

# ══════════════════════════════════════════════════════════════════════════════
# 7.  VaR / CVaR Position Sizing
#     Reference: Ahmed (2023) arXiv:2309.09094
#     "Minimum Variance and CVaR-based sizing directly controls VaR during
#      crisis events where ATR dramatically understates tail risk."
# ══════════════════════════════════════════════════════════════════════════════

# Confidence level for VaR / CVaR computation
VAR_CONFIDENCE = 0.95           # 95% — industry standard for daily risk

# Rolling window for return history used in VaR calculation
VAR_LOOKBACK_DAYS   = 60             # ~3 months of daily returns

# Maximum allowed CVaR as a fraction of portfolio balance
# "If the expected loss in the worst 5% of outcomes exceeds MAX_CVAR_PCT,
#  size the position down until it doesn't."
CVAR_TAIL_PCT         = 0.02     # 2% of balance
MAX_POSITION_PCT     = 0.1     # hard cap: never exceed 25% of balance per trade
MAX_RISK_PER_TRADE_PCT = 0.02   # alias used in main.py risk dashboard

# VaR estimation method:
#   "auto"           → Cornish-Fisher if ≥60 bars, historical if ≥30, parametric otherwise
#   "historical"     → empirical quantile (no distribution assumption)
#   "parametric"     → Gaussian (fast but understates fat tails)
#   "cornish_fisher" → adjusts for skewness + excess kurtosis (best for equities)
VAR_METHOD = "auto"

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Logistic Probability Model                                     ← NEW
#     Reference: Chan (2013) Algorithmic Trading; scikit-learn docs
#     Replaces the Brownian-motion probability estimator that produced 0%
#     output due to numerical edge-cases.
# ══════════════════════════════════════════════════════════════════════════════

# Enable / disable the logistic model gate
# When False the probability gate is bypassed and all crossover signals pass
USE_PROB_MODEL = True

# Number of bars used to train the logistic model (from the tail of df)
# Minimum ~50 required; 200 gives a reliable fit without overfitting
PROB_TRAIN_BARS = 200

# Probability thresholds (as percentages: 55 = 55% = 0.55 fraction)
# A BUY crossover only fires if P(up) ≥ PROB_BUY_THRESHOLD
# A SELL crossover only fires if P(up) ≤ (100 - PROB_SELL_THRESHOLD)
PROB_BUY_THRESHOLD  = 60    # 55% — require slight edge before buying
PROB_SELL_THRESHOLD = 45    # 45% — sell when model leans bearish

# ══════════════════════════════════════════════════════════════════════════════
# 9.  Walk-Forward Validation                                        ← NEW
#     Reference: Deep et al. (2025) arXiv:2512.12924
#     "Rolling window validation prevents lookahead bias that pervades
#      most backtesting research."
# ══════════════════════════════════════════════════════════════════════════════

# Enable walk-forward validation (runs after engine backtest in Phase 5)
USE_WALK_FORWARD = True

# Training window (bars)   — Deep et al. use 252 = 1 trading year
WF_TRAIN_BARS = 252

# Out-of-sample test window (bars) — Deep et al. use 63 = 1 quarter
WF_TEST_BARS  = 63

# Step size (bars) — how far to roll forward per fold; default = test window
# Leave as None to use WF_TEST_BARS
WF_STEP_BARS: int | None = None

# ══════════════════════════════════════════════════════════════════════════════
# 10. Monte Carlo Robustness Test                                    ← NEW
#     Reference: Ahmed (2023) arXiv:2309.09094 | Wang et al. (2026)
#     Shuffles the observed trade sequence N times to show the realistic
#     distribution of equity outcomes independent of entry/exit order.
# ══════════════════════════════════════════════════════════════════════════════

# Enable Monte Carlo backtest analysis (runs after backtest + walk-forward)
USE_BACKTEST_MC = True

# Number of simulation paths (5 000 is the Wang et al. standard)
MC_SIMULATIONS = 10_000

# Number of drawdown simulation paths (2 000 is sufficient for percentiles)
MC_DD_SIMULATIONS = 5_000

# Set an integer for reproducible results; None = different each run
MC_SEED: int | None = None

# Horizon for Monte Carlo price path simulation (days)
MC_HORIZON_DAYS = 30           # e.g. 1 trading month stress horizon

# Maximum acceptable simulated VaR % over horizon for trade approval
MC_ACCEPTABLE_VAR_PCT = 0.10    # 10% — block trades if worse than this

# ══════════════════════════════════════════════════════════════════════════════
# 11. Multi-Ticker Engine                                            ← NEW
# ══════════════════════════════════════════════════════════════════════════════

# Enable multi-ticker scanning mode
# When True: scans all TICKERS, ranks signals, then continues the pipeline
#            on the top-ranked BUY signal (or falls back to SYMBOL if none)
# When False: runs the single-ticker pipeline on SYMBOL only
USE_MULTI_TICKER = True

# Time Delay between API calls (seconds) — keeps Alpha Vantage under 5 req/min
MULTI_TICKER_DELAY_SEC = 1.5

# Minimum bar count to include a ticker in the scan
MULTI_TICKER_MIN_BARS = 50

# Maximum signals to rank and display (top-N table)
MULTI_TICKER_TOP_N = 3

# ══════════════════════════════════════════════════════════════════
# 14. MOO3 Genetic Programming Engine                       ← NEW
#     Reference: Long, Kampouridis & Papastylianou (2026).
#     "Multi-objective GP-based algorithmic trading using
#      directional changes." AI Review 59:39.
# ══════════════════════════════════════════════════════════════════

# Enable loading a pre-trained MOO3 model at main.py startup
# HOW TO RUN python genetic/run_genetic.py --pop 20 --gens 15 (MORE GENS = SLOWER)
# Set to True after running: python genetic/run_genetic.py
USE_MOO3_PLUGIN = True  # flip to True after first training run

# MOO3 engine training parameters
MOO3_POP_SIZE = 50  # population size P (paper default: 50)
MOO3_N_GENS = 50  # number of generations N (paper default: 50)
MOO3_TOURNAMENT = 3  # tournament size k
MOO3_P_CROSSOVER = 0.80  # subtree crossover probability
MOO3_P_MUTATION = 0.10  # point mutation probability
MOO3_MAX_DEPTH = 5  # maximum tree depth

# Modified Sharpe Ratio weights for Pareto front selection
# Must sum to 1.0  (paper uses equal weights; we match user preference)
MOO3_W_TR = 0.40  # Total Return weight
MOO3_W_WR = 0.30  # Win Rate weight
MOO3_W_DD = 0.30  # Max Drawdown penalty weight

# Plugin weight in the ensemble strategy engine
# (DC strategy uses 2.0 per Long et al. results; MOO3 can be higher)
MOO3_PLUGIN_WEIGHT = 2.5
# ──────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# 12. Circuit-Breakers & Stop-Loss
# ══════════════════════════════════════════════════════════════════════════════

# Halt all trading if portfolio equity drops more than X% from peak
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.20   # 20%

# Halt all trading if daily P&L drops more than X% of starting equity
DAILY_LOSS_LIMIT_PCT = 0.05         # 5%

# Per-trade stop-loss: close position if price drops more than X% from entry
USE_STOP_LOSS = True
STOP_LOSS_PCT = 0.05                # 5%

# Per-trade take-profit: close position if price rises more than X% from entry
USE_TAKE_PROFIT = True
TAKE_PROFIT_PCT = 0.08          # 8% target → ~1.6:1 reward:risk with 5% stop

# ══════════════════════════════════════════════════════════════════════════════
# 13. Live Execution Flags
# ══════════════════════════════════════════════════════════════════════════════

# Minimum signal probability required to pass the signal gate (%)
# Uses the probability estimator output (buy_3d_pct / sell_3d_pct scale 0-100)
MIN_SIGNAL_PROBABILITY = 55     # 50% for test 55% = same as PROB_BUY_THRESHOLD

# Run Monte Carlo stress test before each live BUY (live gate, risk_manager.py)
RUN_STRESS_TEST = True