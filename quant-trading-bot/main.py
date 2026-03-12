"""
QuantBot — Multi-Strategy Trading Bot
======================================
Run:
    python main.py

Phases:
    1. Download market data  (Alpha Vantage or simulated fallback)
    2. Generate signals      (strategy selected in settings.py)
    3. Apply filters         (ADX trend filter + OBV volume confirmation)
    4. Backtest              (backtester.py — includes Sharpe + drawdown)
    5. Paper trade           (broker.py → Alpaca)
"""

from data.market_data import get_historical_data
from config.settings import STRATEGY
from strategies import STRATEGY_FACTORY
from backtesting.backtester import backtest
from execution.broker import buy, sell, get_account
from risk.risk_manager import calculate_position_size
from config.settings import (
    SYMBOL,
    SHORT_WINDOW, LONG_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    BB_PERIOD, BB_STD,
    STOCH_K, STOCH_D,
    INITIAL_CAPITAL,
    RISK_PERCENT,
    ALPACA_API_KEY,
    STRATEGY,
    ADX_PERIOD, ADX_THRESHOLD,
    OBV_MA_PERIOD,
    USE_ADX_FILTER,
    USE_OBV_FILTER,
)
from utils.logger import setup_logger

logger = setup_logger()

SEP  = "═" * 66
SEP2 = "─" * 66

def _fmt_pct(v): return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"
def _fmt_usd(v): return f"${v:>12,.2f}"


def print_backtest_results(result: dict):
    strategy_display = {
        "sma":        f"SMA({SHORT_WINDOW}/{LONG_WINDOW})",
        "macd":       f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})",
        "rsi":        f"RSI({RSI_PERIOD}, oversold={RSI_OVERSOLD})",
        "bollinger":  f"Bollinger Bands({BB_PERIOD}, {BB_STD}σ)",
        "stochastic": f"Stochastic({STOCH_K},{STOCH_D})",
    }.get(STRATEGY, STRATEGY.upper())

    filters_active = []
    if USE_ADX_FILTER: filters_active.append(f"ADX>{ADX_THRESHOLD}")
    if USE_OBV_FILTER: filters_active.append(f"OBV(ma={OBV_MA_PERIOD})")
    filter_str = "  filters: " + " + ".join(filters_active) if filters_active else "  no filters"

    logger.info(SEP)
    logger.info(f"  BACKTEST — {SYMBOL}  {strategy_display}")
    logger.info(f"  {filter_str}")
    logger.info(SEP)
    logger.info(f"  {'Initial Capital':<24} {_fmt_usd(INITIAL_CAPITAL)}")
    logger.info(f"  {'Final Value':<24} {_fmt_usd(result['final_value'])}  ← {_fmt_pct(result['return_pct'])}")
    logger.info(SEP2)
    logger.info(f"  {'Buy & Hold':<24} {'':>13}{_fmt_pct(result['buy_hold'])}")
    logger.info(f"  {'Alpha':<24} {'':>13}{_fmt_pct(result['return_pct'] - result['buy_hold'])}")
    logger.info(SEP2)
    # ── NEW risk-adjusted metrics ─────────────────────────────────────────
    sharpe_note = (
        "  (good)" if result['sharpe'] > 1.0 else
        "  (marginal)" if result['sharpe'] > 0.5 else
        "  (poor — consider param tuning)"
    )
    dd_note = (
        "  (comfortable)" if result['max_drawdown'] > -10 else
        "  (moderate)" if result['max_drawdown'] > -20 else
        "  (high — review position sizing)"
    )
    logger.info(f"  {'Sharpe Ratio':<24} {result['sharpe']:>13.2f}{sharpe_note}")
    logger.info(f"  {'Max Drawdown':<24} {result['max_drawdown']:>12.2f}%{dd_note}")
    logger.info(f"  {'Avg Trade Return':<24} {result['avg_trade_pct']:>12.2f}%")
    logger.info(SEP2)
    logger.info(f"  {'Win Rate':<24} {result['win_rate']:>12.1f}%")
    logger.info(f"  {'Total Trades':<24} {result['n_trades']:>13}")
    logger.info(SEP)


def run():
    # ── 1. Market data ────────────────────────────────────────
    logger.info("[ 1 / 5 ]  Downloading market data...")
    df = get_historical_data()
    logger.info(f"           {len(df)} bars loaded for {SYMBOL}")

    # ── 2. Strategy signals ───────────────────────────────────
    logger.info("[ 2 / 5 ]  Generating signals...")
    logger.info(f"           Strategy : {STRATEGY.upper()}")

    generate_signals_func = STRATEGY_FACTORY.get(STRATEGY)
    if generate_signals_func is None:
        logger.error(f"Unknown strategy '{STRATEGY}' — falling back to SMA")
        generate_signals_func = STRATEGY_FACTORY["sma"]

    df = generate_signals_func(df)
    raw_buys  = (df["crossover"] == 1).sum()
    raw_sells = (df["crossover"] == -1).sum()
    logger.info(f"           Raw signals : {raw_buys} BUY · {raw_sells} SELL")

    # ── 3. Apply filters ──────────────────────────────────────
    logger.info("[ 3 / 5 ]  Applying filters...")

    if USE_ADX_FILTER:
        from strategies.adx_filter import apply_adx_filter, add_adx
        df = add_adx(df)
        trending_pct = df["adx_trending"].mean() * 100
        logger.info(f"           ADX filter (>{ADX_THRESHOLD}) : {trending_pct:.0f}% of bars are trending")
        df = apply_adx_filter(df)
    else:
        logger.info("           ADX filter : disabled")

    if USE_OBV_FILTER:
        from strategies.obv_filter import apply_obv_filter, add_obv
        df = add_obv(df)
        df = apply_obv_filter(df)
        logger.info(f"           OBV filter (ma={OBV_MA_PERIOD}) : applied")
    else:
        logger.info("           OBV filter : disabled")

    filtered_buys  = (df["crossover"] == 1).sum()
    filtered_sells = (df["crossover"] == -1).sum()
    suppressed = (raw_buys + raw_sells) - (filtered_buys + filtered_sells)
    logger.info(f"           After filters : {filtered_buys} BUY · {filtered_sells} SELL  "
                f"({suppressed} low-quality signals suppressed)")

    # ── 4. Backtest ───────────────────────────────────────────
    logger.info("[ 4 / 5 ]  Running backtest...")
    result = backtest(df)
    print_backtest_results(result)

    # ── 5. Live signal + paper trade ──────────────────────────
    logger.info("[ 5 / 5 ]  Checking live signal...")
    latest       = df.iloc[-1]
    signal       = latest["signal"]
    crossover    = latest["crossover"]
    latest_price = latest["close"]
    latest_date  = df.index[-1].strftime("%Y-%m-%d")

    regime_label = (
        "BULLISH 🟢" if signal == 1 else
        "BEARISH 🔴" if signal == -1 else
        "NEUTRAL   "
    )

    adx_val  = f"{latest.get('adx', float('nan')):.1f}" if USE_ADX_FILTER else "n/a"
    obv_dir  = ("rising 📈" if latest.get("obv_rising", False) else "falling 📉") if USE_OBV_FILTER else "n/a"

    logger.info(f"           Date          : {latest_date}")
    logger.info(f"           Price         : ${latest_price:.2f}")
    logger.info(f"           Current regime: {regime_label}")
    logger.info(f"           ADX           : {adx_val}  (>{ADX_THRESHOLD} = trending)" if USE_ADX_FILTER
                else f"           ADX           : disabled")
    logger.info(f"           OBV           : {obv_dir}" if USE_OBV_FILTER
                else f"           OBV           : disabled")

    from strategies.probability_estimator import estimate_crossover_probability
    probs = estimate_crossover_probability(df)
    logger.info(f"           Forecast      : {probs['explanation']}")

    alpaca_ready = ALPACA_API_KEY and ALPACA_API_KEY != "YOUR_KEY"

    if alpaca_ready:
        account = get_account()
        cash = float(account.cash) if account else float(INITIAL_CAPITAL)
        logger.info(f"           Alpaca cash   : ${cash:,.2f}")

        if crossover == 1:
            qty = calculate_position_size(
                balance=cash,
                risk_percent=RISK_PERCENT,
                price=latest_price,
            )
            if qty > 0:
                logger.info(f"           ★ BUY — ordering {qty}x {SYMBOL} (ADX+OBV confirmed)")
                buy(SYMBOL, qty)
            else:
                logger.info("           BUY signal but qty=0 (risk limit)")
        elif crossover == -1:
            logger.info(f"           ★ SELL — ordering {SYMBOL}")
            sell(SYMBOL, 1)
        else:
            logger.info("           No crossover today — holding")
    else:
        logger.info("           Alpaca keys not set — skipping live order")

    logger.info(SEP + "\n")


if __name__ == "__main__":
    run()