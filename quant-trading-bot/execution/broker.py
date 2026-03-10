import alpaca_trade_api as tradeapi
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY
from utils.logger import setup_logger

logger = setup_logger()

BASE_URL = "https://paper-api.alpaca.markets"   # swap for live URL when ready

# Initialise once at import time — reused for every order
api = tradeapi.REST(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    BASE_URL,
    api_version="v2",
)


def get_account():
    """Return account details (cash, buying power, etc.)."""
    try:
        return api.get_account()
    except Exception as e:
        logger.error(f"[broker] Could not fetch account: {e}")
        return None


def buy(symbol: str, qty: int):
    """Submit a market buy order."""
    if qty <= 0:
        logger.warning(f"[broker] Skipping BUY — qty={qty} is not valid")
        return
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="gtc",
        )
        logger.info(f"[broker] BUY  {qty}x {symbol} submitted · order id: {order.id}")
        return order
    except Exception as e:
        logger.error(f"[broker] BUY failed: {e}")


def sell(symbol: str, qty: int):
    """Submit a market sell order."""
    if qty <= 0:
        logger.warning(f"[broker] Skipping SELL — qty={qty} is not valid")
        return
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="gtc",
        )
        logger.info(f"[broker] SELL {qty}x {symbol} submitted · order id: {order.id}")
        return order
    except Exception as e:
        logger.error(f"[broker] SELL failed: {e}")
