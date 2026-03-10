import os
from dotenv import load_dotenv

load_dotenv()

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

SYMBOL = os.getenv("SYMBOL", "ACHR")

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
