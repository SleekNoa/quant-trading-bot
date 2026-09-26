"""
Strategy factory — makes it easy to select any strategy from settings.

Note: The walk-forward validation uses individual strategy functions,
while the live engine uses the multi-strategy ensemble via strategy_engine.py.
"""

from .moving_average_strategy import generate_signals as sma_generate_signals
from .rsi_strategy import generate_signals as rsi_generate_signals
from .macd_strategy import generate_signals as macd_generate_signals
from .bollinger_strategy import generate_signals as bollinger_generate_signals
from .stochastic_strategy import generate_signals as stochastic_generate_signals
from .directional_change_strategy import generate_signals as dc_generate_signals

STRATEGY_FACTORY = {
    "sma": sma_generate_signals,
    "rsi": rsi_generate_signals,
    "macd": macd_generate_signals,
    "bollinger": bollinger_generate_signals,
    "stochastic": stochastic_generate_signals,
    "directional_change": dc_generate_signals,
}

# Default strategy for walk-forward when STRATEGY setting is empty or invalid
DEFAULT_STRATEGY = "macd"