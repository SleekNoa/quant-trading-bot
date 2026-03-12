import os
from dotenv import load_dotenv

load_dotenv()

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

SYMBOL = os.getenv("ONDS", "ONDS")

# ── Strategy Selection ────────────────────────────────────────────────
# Change this value to switch strategies without touching main.py
STRATEGY = "macd"          # options: "sma", "rsi", "macd", "bollinger", "stochastic"

# You can keep all parameters here — they are only used by their respective strategy
SHORT_WINDOW = 20
LONG_WINDOW  = 50

RSI_PERIOD      = 14
RSI_OVERSOLD    = 30
RSI_OVERBOUGHT  = 70

MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

BB_PERIOD = 20
BB_STD    = 2.0

STOCH_K = 14
STOCH_D = 3

INITIAL_CAPITAL = 10000
RISK_PERCENT    = 0.01

SHOW_ZSCORE_IN_PROB = True

# ── ADX Trend Filter ──────────────────────────────────────────────────
# ADX measures trend STRENGTH (not direction). Signals are only taken
# when ADX >= ADX_THRESHOLD, meaning the market is actually trending.
# Below 25 = choppy/ranging — momentum indicators produce false signals.
ADX_PERIOD    = 14
ADX_THRESHOLD = 25

# ── OBV Volume Confirmation ───────────────────────────────────────────
# OBV rising = buying pressure confirms the move.
# OBV_MA_PERIOD is the smoothing window for OBV trend detection.
OBV_MA_PERIOD = 5

# ── Filter Flags (set False to disable either filter) ────────────────
USE_ADX_FILTER = True
USE_OBV_FILTER = True

# ── Data Source ───────────────────────────────────────────────────────
# True  = always use 500-bar synthetic random-walk (tests strategy logic)
# False = use live Alpha Vantage data (real OHLCV, ~100 bars free tier)
# NOTE: Simulated data is NOT real market data. Use it to test that the
# strategy and filters behave correctly over a full cycle, not to predict
# real performance.
USE_SIMULATED_DATA = False