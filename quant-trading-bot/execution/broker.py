"""
execution/broker.py — Alpaca Paper Trading Execution
=====================================================
Fixes vs original:
    1. Uses alpaca-py (current SDK) instead of deprecated alpaca-trade-api
    2. Checks existing position before buying (prevents double-entry)
    3. Sells ACTUAL position qty (not hardcoded 1)
    4. Submits bracket orders with stop-loss when USE_STOP_LOSS=True
    5. Returns structured result dicts for downstream logging
"""

import os
from utils.logger import logger
from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    USE_STOP_LOSS, STOP_LOSS_PCT,
    USE_TAKE_PROFIT, TAKE_PROFIT_PCT,
)

# ── SDK import with graceful fallback ─────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest,
        StopLossRequest, TakeProfitRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    ALPACA_PY_AVAILABLE = True
except ImportError:
    try:
        import alpaca_trade_api as tradeapi
        ALPACA_PY_AVAILABLE = False
        logger.warning("[broker] alpaca-py not found — falling back to alpaca-trade-api. "
                       "Install alpaca-py for best compatibility: pip install alpaca-py")
    except ImportError:
        ALPACA_PY_AVAILABLE = None
        logger.error("[broker] No Alpaca SDK found. Install: pip install alpaca-py")

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def _get_client():
    """Return an Alpaca TradingClient (alpaca-py) or legacy REST client."""
    if ALPACA_PY_AVAILABLE is True:
        return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    elif ALPACA_PY_AVAILABLE is False:
        return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY,
                             PAPER_BASE_URL, api_version="v2")
    return None


def get_account() -> dict | None:
    """
    Return account info: cash, portfolio_value, buying_power.
    Returns a plain dict regardless of SDK version.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        if ALPACA_PY_AVAILABLE:
            acct = client.get_account()
            return {
                "cash":            float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "buying_power":    float(acct.buying_power),
                "equity":          float(acct.equity),
            }
        else:
            acct = client.get_account()
            return {
                "cash":            float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "buying_power":    float(acct.buying_power),
                "equity":          float(acct.equity),
            }
    except Exception as e:
        logger.error(f"[broker] get_account failed: {e}")
        return None


def get_position(symbol: str) -> dict | None:
    """
    Return current open position for symbol, or None if flat.
    Returns dict with: qty, avg_entry_price, market_value, unrealized_pl
    """
    client = _get_client()
    if client is None:
        return None
    try:
        if ALPACA_PY_AVAILABLE:
            pos = client.get_open_position(symbol)
            return {
                "qty":             int(float(pos.qty)),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value":    float(pos.market_value),
                "unrealized_pl":   float(pos.unrealized_pl),
            }
        else:
            pos = client.get_position(symbol)
            return {
                "qty":             int(float(pos.qty)),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value":    float(pos.market_value),
                "unrealized_pl":   float(pos.unrealized_pl),
            }
    except Exception:
        # Position doesn't exist = we're flat
        return None


def buy(symbol: str, qty: int, stop_loss_price: float = None) -> dict | None:
    """
    Submit a market BUY order.

    If USE_STOP_LOSS is True and stop_loss_price is provided, submits a
    bracket order with an attached stop-loss leg.

    Guards:
        - skips if qty <= 0
        - skips if an open position already exists (prevents double-entry)
    """
    if qty <= 0:
        logger.warning(f"[broker] Skipping BUY — qty={qty} invalid")
        return None

    # Guard: check for existing position
    existing = get_position(symbol)
    if existing and existing["qty"] > 0:
        logger.warning(
            f"[broker] Skipping BUY — already holding {existing['qty']}x {symbol}. "
            f"Close position first or wait for SELL signal."
        )
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        if ALPACA_PY_AVAILABLE:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            # Attach stop-loss if configured and price is provided
            if USE_STOP_LOSS and stop_loss_price:
                order_data.stop_loss = StopLossRequest(stop_price=round(stop_loss_price, 2))

            order = client.submit_order(order_data)
            order_id = order.id
        else:
            # Legacy SDK (alpaca-trade-api)
            kwargs = dict(
                symbol=symbol, qty=qty, side="buy",
                type="market", time_in_force="gtc"
            )
            if USE_STOP_LOSS and stop_loss_price:
                kwargs["order_class"] = "bracket"
                kwargs["stop_loss"]   = {"stop_price": round(stop_loss_price, 2)}
            order = client.submit_order(**kwargs)
            order_id = order.id

        stop_note = f" | stop-loss @ ${stop_loss_price:.2f}" if (USE_STOP_LOSS and stop_loss_price) else ""
        logger.info(f"[broker] ✅ BUY  {qty}x {symbol}  submitted{stop_note}  · id: {order_id}")
        return {"order_id": str(order_id), "qty": qty, "symbol": symbol, "side": "buy"}

    except Exception as e:
        logger.error(f"[broker] BUY failed: {e}")
        return None


def sell(symbol: str, qty: int = None) -> dict | None:
    """
    Submit a market SELL order.

    If qty is None (default), sells the ENTIRE current position.
    This fixes the original bug where only 1 share was ever sold.
    """
    client = _get_client()
    if client is None:
        return None

    # Determine qty from actual position if not specified
    if qty is None:
        position = get_position(symbol)
        if position is None or position["qty"] <= 0:
            logger.warning(f"[broker] Skipping SELL — no open position in {symbol}")
            return None
        qty = position["qty"]
        logger.info(f"[broker] Selling full position: {qty}x {symbol}")

    if qty <= 0:
        logger.warning(f"[broker] Skipping SELL — qty={qty} invalid")
        return None

    try:
        if ALPACA_PY_AVAILABLE:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
            order = client.submit_order(order_data)
            order_id = order.id
        else:
            order = client.submit_order(
                symbol=symbol, qty=qty, side="sell",
                type="market", time_in_force="gtc"
            )
            order_id = order.id

        logger.info(f"[broker] ✅ SELL {qty}x {symbol}  submitted  · id: {order_id}")
        return {"order_id": str(order_id), "qty": qty, "symbol": symbol, "side": "sell"}

    except Exception as e:
        logger.error(f"[broker] SELL failed: {e}")
        return None


def close_position(symbol: str) -> dict | None:
    """Convenience: close entire position immediately."""
    return sell(symbol, qty=None)