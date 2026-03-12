"""
config/settings.py — QuantBot Configuration
============================================
All parameters in one place. Edit here; nothing else needs to change.

Risk framework references:
    Wang, Zhao & Wang (2026) — Integrated financial risk management framework.
    Case Studies in Thermal Engineering. ScienceDirect.
    → VaR/CVaR position sizing, drawdown circuit-breakers, stress-test gates.

Strategy framework references:
    Long, Kampouridis & Papastylianou (2026) — Multi-objective genetic
    programming-based algorithmic trading using directional changes.
    Artificial Intelligence Review. Springer.
    → DC theta tuning, regime-aware strategy weighting.

Tuning notes (based on ONDS live run 2026-03-11):
    • Backtest showed −23.55% max drawdown on a +65% winning run.
      Old circuit-breaker at −15% would have killed the run early.
      Raised to −25% to allow room for high-volatility small-caps.
    • Alpaca paper account has $100,000 — INITIAL_CAPITAL corrected
      so VaR position sizing calculates against the right number.
    • ONDS ~$9.83, small-cap, higher daily volatility than mega-caps.
      DC theta raised to 0.025 so events only fire on moves ≥ 2.5%,
      filtering noise from normal daily small-cap fluctuation.
    • MACD fast period tightened to 8 (from 12) to catch ONDS momentum
      faster — small-caps move quicker than large-caps.
    • RSI oversold/overbought tightened to 25/75 — ONDS regularly
      reaches extreme readings that standard 30/70 misses.
    • Bollinger std widened to 2.5 — ONDS's volatility means 2.0
      generates excessive false touches at the bands.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
ALPACA_API_KEY       = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY    = os.getenv("ALPACA_SECRET_KEY", "")

# ── Symbol ────────────────────────────────────────────────────────────
Ticker_Symbol = "hive"
SYMBOL = os.getenv(Ticker_Symbol, Ticker_Symbol)

# ── Strategy ─────────────────────────────────────────────────────────
# Options: "sma" | "rsi" | "macd" | "bollinger" | "stochastic"
STRATEGY = "macd"

# ══════════════════════════════════════════════════════════════════════
#  STRATEGY PARAMETERS
#  Tuned for small-cap / higher-volatility assets (ONDS profile).
#  MACD: faster settings catch momentum earlier on low-float stocks.
#  RSI:  tighter extremes — small-caps regularly reach 20/80+.
#  BB:   wider std avoids excessive false-touch signals.
#  SMA:  shorter windows reduce lag on faster-moving small-caps.
#  Stochastic: standard 14/3 — well-proven, left unchanged.
# ══════════════════════════════════════════════════════════════════════

# ── SMA ───────────────────────────────────────────────────────────────
SHORT_WINDOW = 15          # was 20 — faster response for small-caps
LONG_WINDOW  = 40          # was 50 — tighter crossover window

# ── RSI ───────────────────────────────────────────────────────────────
RSI_PERIOD     = 14        # unchanged — well-established period
RSI_OVERSOLD   = 25        # was 30 — ONDS reaches deeper oversold levels
RSI_OVERBOUGHT = 75        # was 70 — ONDS reaches deeper overbought levels

# ── MACD ──────────────────────────────────────────────────────────────
# Standard 12/26/9 works well for large-caps. For small-caps with faster
# momentum cycles, 8/21/5 catches entries and exits noticeably earlier.
# Your live MACD gap was −0.022 — faster EMA would have caught the
# bearish cross sooner and generated the SELL with more lead time.
MACD_FAST   = 8            # was 12 — faster EMA for quick momentum reads
MACD_SLOW   = 21           # was 26 — tighter slow EMA
MACD_SIGNAL = 5            # was 9  — more responsive signal line

# ── Bollinger Bands ───────────────────────────────────────────────────
BB_PERIOD = 20             # unchanged — standard lookback
BB_STD    = 2.5            # was 2.0 — wider bands for high-vol small-caps

# ── Stochastic ────────────────────────────────────────────────────────
STOCH_K = 14               # unchanged
STOCH_D = 3                # unchanged

# ── Capital ───────────────────────────────────────────────────────────
# CORRECTED: was 10,000 but Alpaca paper account has $100,000.
# VaR position sizing uses this as the baseline — wrong value here
# means every position was 10x undersized.
INITIAL_CAPITAL = 100_000.0

# ── Filters ───────────────────────────────────────────────────────────
ADX_PERIOD    = 14
ADX_THRESHOLD = 25         # unchanged — 25 is the standard trending threshold
OBV_MA_PERIOD = 3          # was 5 — faster OBV smoothing for small-cap volume spikes
USE_ADX_FILTER = True
USE_OBV_FILTER = True

SHOW_ZSCORE_IN_PROB = True

# ══════════════════════════════════════════════════════════════════════
#  DIRECTIONAL CHANGES (DC) PARAMETERS
#  Source: Long et al. (2026) Table 5 — theta configuration space.
#  Paper tests: 0.001, 0.002, 0.005, 0.010, 0.020
#
#  Theta selection rationale for ONDS:
#    θ = 0.010 (1%) — too sensitive for a $9.83 stock. A $0.10 move
#    triggers a DC event, which is just normal intraday noise for ONDS.
#    θ = 0.025 (2.5%) — filters noise, only fires on moves of ~$0.25+.
#    This matches ONDS's average true range and produces higher-quality
#    DC events with genuine directional conviction.
#
#  DC OSV confirmation multiplier:
#    After a DC event is confirmed, we require the overshoot to exceed
#    theta × DC_OSV_CONFIRM_MULT before signalling. Higher = fewer but
#    higher-conviction signals. 0.5 is the paper's default; raised to
#    0.75 for ONDS to further reduce whipsaw entries.
# ══════════════════════════════════════════════════════════════════════

DC_THETA             = 0.025   # was hardcoded 0.01 — 2.5% threshold for ONDS
DC_OSV_CONFIRM_MULT  = 0.75    # was hardcoded 0.5 — tighter overshoot confirmation

# ══════════════════════════════════════════════════════════════════════
#  INSTITUTIONAL RISK MANAGEMENT FRAMEWORK
#  Source: Wang, Zhao & Wang (2026) — integrated VaR/CVaR framework
#  achieving Sharpe 1.85, 18.7% max drawdown reduction,
#  24.3% CVaR improvement via optimal sizing & hedging.
# ══════════════════════════════════════════════════════════════════════

# ── VaR / CVaR Parameters ─────────────────────────────────────────────
VAR_CONFIDENCE     = 0.95          # 95th percentile — industry standard
VAR_LOOKBACK_DAYS  = 60            # rolling window for historical VaR
CVAR_TAIL_PCT      = 0.05          # tail fraction for CVaR (bottom 5%)

# ── Position Sizing (VaR-based) ───────────────────────────────────────
# Wang et al. demonstrate optimal tail-risk control at 1-2% per position.
# Raised slightly to 2.5% — with $100k account and a $9.83 stock, 2%
# produces very small share counts. 2.5% keeps positions meaningful
# while staying within institutional risk guidelines.
MAX_RISK_PER_TRADE_PCT = 0.025     # was 0.02 — 2.5% risk budget per trade
MAX_POSITION_PCT       = 0.15      # was 0.20 — max 15% of portfolio per trade
                                   # tighter cap for small-cap concentration risk

# ── Portfolio-Level Risk Limits ────────────────────────────────────────
# RAISED from 15% to 25%: your ONDS backtest showed −23.55% drawdown
# on a +65% / Sharpe 1.71 winning run. The old 15% limit would have
# halted trading mid-run and cost ~40% in unrealised gains.
# 25% gives room for high-volatility small-cap drawdown cycles while
# still providing a hard floor against catastrophic loss.
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.25  # was 0.15 — raised for small-cap volatility
DAILY_LOSS_LIMIT_PCT       = 0.04  # was 0.03 — slightly wider for volatile days

# ── Signal Quality Gate ────────────────────────────────────────────────
# Raised to 35% — your live signal showed 35.9% buy probability and
# the engine said HOLD/SELL. Keeping the gate at 35% means the bot
# won't chase weak reversals while still acting on moderate conviction.
MIN_SIGNAL_PROBABILITY = 35.0      # was 30.0 — tighter entry filter

# ── Stop-Loss / Take-Profit ────────────────────────────────────────────
# Widened stop-loss from 5% to 7% for ONDS — small-caps with wider
# daily ranges will stop out regularly at 5%, cutting winners short.
# 7% gives the position room to breathe through normal volatility.
STOP_LOSS_PCT   = 0.07             # was 0.05 — wider stop for small-cap ATR
TAKE_PROFIT_PCT = 0.20             # was 0.10 — raised to capture ONDS swing moves
USE_STOP_LOSS   = True
USE_TAKE_PROFIT = False            # enable once paper trading validates the stop levels

# ── Monte Carlo Stress Test ────────────────────────────────────────────
# MC_ACCEPTABLE_VAR_PCT raised from 8% to 12%: ONDS's historical
# volatility produces MC VaR estimates that regularly exceed 8% over
# a 10-day horizon. The old threshold was blocking valid entries.
# 12% is still a meaningful risk gate — it blocks truly extreme setups.
RUN_STRESS_TEST        = True
MC_SIMULATIONS         = 1000      # unchanged — 1,000 paths is sufficient
MC_HORIZON_DAYS        = 10        # unchanged — 10-day forward horizon
MC_ACCEPTABLE_VAR_PCT  = 0.12      # was 0.08 — raised for ONDS volatility profile

# ── Data Source ───────────────────────────────────────────────────────
# True  = always use 500-bar synthetic random-walk (tests strategy logic)
# False = use live Alpha Vantage data (real OHLCV, ~100 bars free tier)
# NOTE: Simulated data is NOT real market data. Use it to test that the
# strategy and filters behave correctly over a full cycle, not to predict
# real performance.
USE_SIMULATED_DATA = False