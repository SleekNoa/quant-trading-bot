"""
execution/broker.py — Alpaca Paper Trading Execution
=====================================================
Fixes vs original:
    1. Uses alpaca-py (current SDK) instead of deprecated alpaca-trade-api
    2. Checks existing position before buying (prevents double-entry)
    3. Sells ACTUAL position qty (not hardcoded 1)
    4. Submits bracket orders with stop-loss when USE_STOP_LOSS=True
    5. Returns structured result dicts for downstream logging
    6. get_today_activity reads filled orders — not position qty
    7. sell() uses OrderSide.SELL (was incorrectly OrderSide.BUY)
    8. sell() no longer references undefined stop_loss_price / price
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
        logger.warning("[broker] alpaca-py not found — falling back to alpaca-trade-api.")
    except ImportError:
        ALPACA_PY_AVAILABLE = None
        logger.error("[broker] No Alpaca SDK found. Install: pip install alpaca-py")

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def _get_client():
    if ALPACA_PY_AVAILABLE is True:
        return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    elif ALPACA_PY_AVAILABLE is False:
        return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY,
                             PAPER_BASE_URL, api_version="v2")
    return None


def get_account() -> dict | None:
    client = _get_client()
    if client is None:
        return None
    try:
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
    client = _get_client()
    if client is None:
        return None
    try:
        if ALPACA_PY_AVAILABLE:
            pos = client.get_open_position(symbol)
        else:
            pos = client.get_position(symbol)
        return {
            "qty":             int(float(pos.qty)),
            "avg_entry_price": float(pos.avg_entry_price),
            "market_value":    float(pos.market_value),
            "unrealized_pl":   float(pos.unrealized_pl),
        }
    except Exception:
        return None


def buy(symbol: str, qty: int, stop_loss_price: float = None) -> dict | None:
    """
    Submit a market BUY order with optional stop-loss bracket.
    Guards: skips if qty <= 0 or an open position already exists.
    """
    if qty <= 0:
        logger.warning(f"[broker] Skipping BUY — qty={qty} invalid")
        return None

    existing = get_position(symbol)
    if existing and existing["qty"] > 0:
        logger.warning(
            f"[broker] Skipping BUY — already holding {existing['qty']}x {symbol}."
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
            if USE_STOP_LOSS and stop_loss_price:
                order_data.stop_loss = StopLossRequest(
                    stop_price=round(stop_loss_price, 2)
                )
            order = client.submit_order(order_data)
            order_id = order.id
        else:
            kwargs = dict(symbol=symbol, qty=qty, side="buy",
                          type="market", time_in_force="gtc")
            if USE_STOP_LOSS and stop_loss_price:
                kwargs["order_class"] = "bracket"
                kwargs["stop_loss"]   = {"stop_price": round(stop_loss_price, 2)}
            order = client.submit_order(**kwargs)
            order_id = order.id

        stop_note = (f" | stop-loss @ ${stop_loss_price:.2f}"
                     if (USE_STOP_LOSS and stop_loss_price) else "")
        logger.info(f"[broker] ✅ BUY  {qty}x {symbol}  submitted{stop_note}  · id: {order_id}")
        return {"order_id": str(order_id), "qty": qty, "symbol": symbol, "side": "buy"}

    except Exception as e:
        logger.error(f"[broker] BUY failed: {e}")
        return None


def sell(symbol: str, qty: int = None) -> dict | None:
    """
    Submit a market SELL order.
    If qty is None, sells the entire current position.
    """
    client = _get_client()
    if client is None:
        return None

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
                side=OrderSide.SELL,          # Bug 3 fix: was OrderSide.BUY
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


def get_today_activity(symbol: str) -> dict:
    """
    Return today's fill activity and pending orders for symbol.
    bought_qty reflects FILLED buy orders today — not held position qty.
    """
    result = {
        "bought_qty":         0,
        "sold_qty":           0,
        "net_qty":            0,
        "pending_buy_count":  0,
        "pending_sell_count": 0,
        "has_open_position":  False,
    }

    # Open position flag — for reference only, does NOT set bought_qty
    position     = get_position(symbol)
    has_position = position is not None and position["qty"] > 0
    result["has_open_position"] = has_position

    client = _get_client()
    if client is None:
        logger.warning("[broker] get_today_activity: no client")
        return result

    try:
        if ALPACA_PY_AVAILABLE:
            import datetime, pytz
            tz    = pytz.timezone("America/New_York")
            now   = datetime.datetime.now(tz)
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end   = now.replace(hour=23, minute=59, second=59, microsecond=0)

            logger.info(f"[broker] Scanning open orders for {symbol} ({start.date()})")

            # ── Pending orders ────────────────────────────────────────────────
            open_req = GetOrdersRequest(
                status="open",
                after=start.isoformat(),
                until=end.isoformat(),
                limit=500,
            )
            all_open = client.get_orders(open_req)
            logger.info(f"[broker] Open orders returned: {len(all_open)}")

            for o in all_open:
                if str(o.symbol) != symbol:
                    continue
                side = str(o.side).lower()
                if "buy"  in side: result["pending_buy_count"]  += 1
                if "sell" in side: result["pending_sell_count"] += 1

            # ── Today's filled orders (source of truth for bought_qty) ────────
            filled_req = GetOrdersRequest(
                status="closed",
                after=start.isoformat(),
                until=end.isoformat(),
                limit=500,
            )
            all_filled = client.get_orders(filled_req)

            for o in all_filled:
                if str(o.symbol) != symbol:
                    continue
                side = str(o.side).lower()
                qty  = int(float(o.filled_qty or 0))
                if "buy"  in side: result["bought_qty"] += qty
                if "sell" in side: result["sold_qty"]   += qty

        result["net_qty"] = result["bought_qty"] - result["sold_qty"]
        logger.info(f"[broker] Activity for {symbol}: {result}")

    except Exception as e:
        logger.error(f"[broker] get_today_activity scan failed: {e}", exc_info=True)

    return result


def close_position(symbol: str) -> dict | None:
    """Convenience: close entire position immediately."""
    return sell(symbol, qty=None)