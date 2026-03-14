"""
execution/exit_monitor.py — Dedicated Position Exit Layer
==========================================================
Implements the three exit rules described in Long, Kampouridis &
Papastylianou (2026) and used internally by the genetic engine
(genetic/fitness.py), now applied to the live paper-trading pipeline.

    "Our strategy for selling an asset is to sell either after n days
     have already passed from the initial purchase, or when a price
     increase of r% has occurred, whichever comes first."
     — Long et al. (2026), Section 4

Rules (checked in priority order)
----------------------------------
    1. Take-profit    — P&L ≥ TAKE_PROFIT_PCT from entry price
    2. Trailing stop  — price drops TRAILING_STOP_PCT from peak since entry
    3. Time-based     — position held ≥ EXIT_MAX_HOLD_DAYS calendar days

This module is completely independent of the strategy engine.
It runs BEFORE the engine decision is acted on in main.py so that
exit logic can never be blocked by a BUY-biased vote.

Entry date resolution
---------------------
    Alpaca's get_position() returns avg_entry_price but NOT the entry date.
    We query filled orders (last 90 days) to find the most recent filled
    BUY order for the symbol, then use its filled_at timestamp.
    Falls back gracefully if order history is unavailable.

Peak price resolution
---------------------
    To compute trailing stop, we need the highest close since entry.
    We slice df from the entry date forward and take df["close"].max().
    This is accurate because df is already loaded in main.py.
"""

from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd

from utils.logger import logger
from config.settings import (
    USE_TAKE_PROFIT,   TAKE_PROFIT_PCT,
    USE_TRAILING_STOP, TRAILING_STOP_PCT,
    USE_TIME_EXIT,     EXIT_MAX_HOLD_DAYS,
    USE_STOP_LOSS,     STOP_LOSS_PCT,
    ALPACA_API_KEY,    ALPACA_SECRET_KEY,
)

# ── SDK import (mirrors broker.py pattern) ────────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    _ALPACA_PY = True
except ImportError:
    try:
        import alpaca_trade_api as tradeapi  # noqa
        _ALPACA_PY = False
    except ImportError:
        _ALPACA_PY = None


def _get_client():
    if _ALPACA_PY is True:
        return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    if _ALPACA_PY is False:
        import alpaca_trade_api as tradeapi
        return tradeapi.REST(
            ALPACA_API_KEY, ALPACA_SECRET_KEY,
            "https://paper-api.alpaca.markets", api_version="v2"
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  1. Entry info resolver
# ══════════════════════════════════════════════════════════════════════════════

def get_entry_info(symbol: str) -> dict:
    """
    Query Alpaca filled orders to find the most recent BUY fill for symbol.

    Returns
    -------
    dict with keys:
        entry_date  : datetime.date | None
        entry_price : float | None
        days_held   : int   (0 if entry_date unknown)
        source      : str   ("alpaca_orders" | "unavailable")
    """
    result = {
        "entry_date":  None,
        "entry_price": None,
        "days_held":   0,
        "source":      "unavailable",
    }

    client = _get_client()
    if client is None:
        logger.warning("[exit] No Alpaca client — entry date unknown")
        return result

    try:
        import pytz
        tz    = pytz.timezone("America/New_York")
        now   = datetime.datetime.now(tz)
        since = now - datetime.timedelta(days=90)

        if _ALPACA_PY:
            req = GetOrdersRequest(
                status="closed",
                after=since.isoformat(),
                limit=200,
            )
            orders = client.get_orders(req)
        else:
            orders = client.list_orders(
                status="filled",
                after=since.isoformat(),
                limit=200,
            )

        # Filter: filled BUY orders for our symbol, newest first
        buy_fills = []
        for o in orders:
            sym  = str(o.symbol)
            side = str(o.side).lower()
            stat = str(o.status).lower()
            if sym == symbol and "buy" in side and "fill" in stat:
                filled_at = getattr(o, "filled_at", None)
                fill_price = getattr(o, "filled_avg_price", None)
                if filled_at:
                    buy_fills.append((filled_at, fill_price))

        if not buy_fills:
            logger.info(f"[exit] No filled BUY orders found for {symbol} in last 90 days")
            return result

        # Most recent fill
        buy_fills.sort(key=lambda x: x[0], reverse=True)
        filled_at, fill_price = buy_fills[0]

        # Normalise to date
        if hasattr(filled_at, "date"):
            entry_date = filled_at.date()
        else:
            entry_date = pd.Timestamp(filled_at).date()

        days_held = (datetime.date.today() - entry_date).days

        result.update({
            "entry_date":  entry_date,
            "entry_price": float(fill_price) if fill_price else None,
            "days_held":   days_held,
            "source":      "alpaca_orders",
        })
        logger.info(
            f"[exit] Entry info for {symbol}: "
            f"date={entry_date}  price=${result['entry_price']}  "
            f"days_held={days_held}"
        )

    except Exception as exc:
        logger.warning(f"[exit] Could not retrieve entry info: {exc}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  2. Peak price since entry (for trailing stop)
# ══════════════════════════════════════════════════════════════════════════════

def get_peak_price_since_entry(
    df: pd.DataFrame,
    entry_date: Optional[datetime.date],
) -> Optional[float]:
    """
    Return the highest closing price from entry_date to the latest bar.
    Uses the already-loaded price history — no additional API call needed.
    """
    if entry_date is None or df is None or df.empty:
        return None

    try:
        mask = df.index.date >= entry_date
        if not mask.any():
            return None
        peak = float(df.loc[mask, "close"].max())
        logger.info(f"[exit] Peak close since {entry_date}: ${peak:.2f}")
        return peak
    except Exception as exc:
        logger.warning(f"[exit] Could not compute peak price: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Exit rule evaluator (UPDATED — now 100% identical priority to backtester)
# ══════════════════════════════════════════════════════════════════════════════
def check_exit_rules(
    position: dict,
    entry_info: dict,
    current_price: float,
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Evaluate all exit rules against the current position.
    Returns the exit reason string if an exit should fire, else None.

    PRIORITY ORDER (exact match to _run_single_backtest + updated MOO3 backtest):
    1. Hard Stop-Loss
    2. Time-based exit
    3. Trailing stop
    4. Take-profit

    This guarantees the live paper-trading exits are identical to what
    the ensemble backtester (including MOO3) simulated.
    """
    entry_price = position.get("avg_entry_price") or entry_info.get("entry_price")
    if not entry_price or entry_price <= 0:
        logger.warning("[exit] Cannot evaluate rules — entry price unknown")
        return None

    pnl_pct = (current_price - entry_price) / entry_price * 100
    days_held = entry_info.get("days_held", 0)
    exit_reason = None

    # 1. Hard Stop-Loss
    if USE_STOP_LOSS:
        sl_threshold = STOP_LOSS_PCT * 100
        if pnl_pct <= sl_threshold:
            exit_reason = f"stop_loss ({pnl_pct:.2f}%)"
            logger.info(f"[exit] ✅ {exit_reason}")
            return exit_reason
        logger.info(f"[exit] Stop-loss: {pnl_pct:+.2f}% (trigger at {sl_threshold:.2f}%)")

    # 2. Time-based exit
    if USE_TIME_EXIT:
        if days_held >= EXIT_MAX_HOLD_DAYS:
            exit_reason = f"time_exit ({days_held} days held)"
            logger.info(f"[exit] ✅ {exit_reason}")
            return exit_reason
        days_rem = EXIT_MAX_HOLD_DAYS - days_held
        logger.info(f"[exit] Time exit: {days_held}/{EXIT_MAX_HOLD_DAYS} days ({days_rem} remaining)")

    # 3. Trailing stop (now before TP — matches backtester)
    if USE_TRAILING_STOP and pnl_pct > 2:
        peak = get_peak_price_since_entry(df, entry_info.get("entry_date"))
        if peak and peak > 0:
            dd_from_peak = (current_price - peak) / peak * 100
            trail_thresh = -TRAILING_STOP_PCT * 100
            if dd_from_peak <= trail_thresh:
                exit_reason = f"trailing_stop ({dd_from_peak:.2f}% from peak ${peak:.2f})"
                logger.info(f"[exit] ✅ {exit_reason}")
                return exit_reason
            logger.info(
                f"[exit] Trailing stop: {dd_from_peak:+.2f}% from peak "
                f"(trigger at {trail_thresh:.1f}%)"
            )
        else:
            logger.info("[exit] Trailing stop: peak price unavailable")

    # 4. Take-profit (last in priority)
    if USE_TAKE_PROFIT:
        tp_threshold = TAKE_PROFIT_PCT * 100
        if pnl_pct >= tp_threshold:
            exit_reason = f"take_profit ({pnl_pct:.2f}%)"
            logger.info(f"[exit] ✅ {exit_reason}")
            return exit_reason
        logger.info(f"[exit] Take-profit: {pnl_pct:+.2f}% (need +{tp_threshold:.1f}%)")

    return None

# ══════════════════════════════════════════════════════════════════════════════
#  4. Main entry point (called from main.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_exit_monitor(
    symbol:        str,
    current_price: float,
    df:            pd.DataFrame,
    position:      Optional[dict] = None,
) -> bool:
    """
    Check all exit rules for an open position and sell if any fire.

    Called at the top of Phase 6 in main.py, BEFORE the engine decision
    is acted on. This ensures exits are never blocked by a BUY-biased vote.

    Parameters
    ----------
    symbol        : active trading symbol
    current_price : latest close price
    df            : full price + indicator DataFrame (already loaded)
    position      : pre-fetched position dict, or None to fetch here

    Returns
    -------
    True  — an exit was triggered and sell() was called  → main.py should return
    False — no exit rule fired  → main.py continues normally
    """
    from execution.broker import get_position, sell as broker_sell

    # Fetch position if not provided
    if position is None:
        position = get_position(symbol)

    if not position or position.get("qty", 0) <= 0:
        logger.info(f"[exit] No open position in {symbol} — exit monitor skipped")
        return False

    qty         = position["qty"]
    entry_price = position.get("avg_entry_price", 0)
    unreal_pl   = position.get("unrealized_pl", 0)

    logger.info(
        f"[exit] Monitoring position: {qty}x {symbol}  "
        f"entry=${entry_price:.2f}  current=${current_price:.2f}  "
        f"unrealized P&L=${unreal_pl:+,.2f}"
    )

    # Resolve entry date from Alpaca order history
    entry_info = get_entry_info(symbol)

    # Evaluate all rules
    exit_reason = check_exit_rules(position, entry_info, current_price, df)

    if exit_reason:
        logger.warning(
            f"[exit] 🔔 EXIT TRIGGERED for {qty}x {symbol}  —  {exit_reason}"
        )
        result = broker_sell(symbol)
        if result:
            logger.info(
                f"[exit] ✅ Sell order submitted  "
                f"(qty={qty}  price≈${current_price:.2f})"
            )
        else:
            logger.error("[exit] ❌ Sell order failed — check broker logs")
        return True

    logger.info(f"[exit] No exit rules triggered for {symbol} — holding")
    return False