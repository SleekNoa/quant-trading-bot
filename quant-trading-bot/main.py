"""
QuantBot — Moving Average Crossover Trading Bot
================================================
Run:
    python main.py

Phases:
    1. Download market data  (Alpha Vantage or simulated fallback)
    2. Generate MA signals   (strategies/moving_average_strategy.py)
    3. Backtest              (backtesting/backtester.py)
    4. Paper trade           (execution/broker.py → Alpaca)
"""

from data.market_data import get_historical_data
# from strategies.moving_average_strategy import generate_signals
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
)
from utils.logger import setup_logger

logger = setup_logger()

# ─────────────────────────────────────────────────────────────
SEP  = "═" * 62
SEP2 = "─" * 62

def _fmt_pct(v):  return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"
def _fmt_usd(v):  return f"${v:>12,.2f}"


def print_backtest_results(result: dict):
    from config.settings import STRATEGY, SHORT_WINDOW, LONG_WINDOW  # add if not already imported

    strategy_display = {
        "sma":        f"SMA({SHORT_WINDOW}/{LONG_WINDOW})",
        "macd":       f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})",
        "rsi":        f"RSI({RSI_PERIOD}, oversold={RSI_OVERSOLD})",
        "bollinger":  f"Bollinger Bands({BB_PERIOD}, {BB_STD}σ)",
        "stochastic": f"Stochastic({STOCH_K},{STOCH_D})",
    }.get(STRATEGY, STRATEGY.upper())

    logger.info(SEP)
    logger.info(f"  BACKTEST — {SYMBOL}  {strategy_display}")
    logger.info(SEP)
    logger.info(f"  {'Initial Capital':<22} {_fmt_usd(INITIAL_CAPITAL)}")
    logger.info(f"  {'Final Value':<22} {_fmt_usd(result['final_value'])}  ← {_fmt_pct(result['return_pct'])}")
    logger.info(f"  {'Buy & Hold':<22} {'':>13}{_fmt_pct(result['buy_hold'])}")
    logger.info(f"  {'Alpha':<22} {'':>13}{_fmt_pct(result['return_pct'] - result['buy_hold'])}")
    logger.info(f"  {'Win Rate':<22} {result['win_rate']:>12.1f}%")
    logger.info(f"  {'Total Trades':<22} {result['n_trades']:>13}")
    logger.info(SEP)


def run():
    # ── 1. Market data ────────────────────────────────────────
    logger.info("[ 1 / 4 ]  Downloading market data...")
    df = get_historical_data()
    logger.info(f"           {len(df)} bars loaded for {SYMBOL}")

    # ── 2. Strategy signals ───────────────────────────────────
    logger.info("[ 2 / 4 ]  Generating MA signals...")
    logger.info(f"           Using strategy: {STRATEGY.upper()}")

    # Get the correct generate_signals function based on settings
    generate_signals_func = STRATEGY_FACTORY.get(STRATEGY)

    if generate_signals_func is None:
        logger.error(f"Unknown strategy '{STRATEGY}' in settings.py")
        logger.error(f"Valid options are: {', '.join(STRATEGY_FACTORY.keys())}")
        logger.warning("Falling back to default 'sma' strategy")
        generate_signals_func = STRATEGY_FACTORY["sma"]

    df = generate_signals_func(df)

    buys = (df["crossover"] == 1).sum()
    sells = (df["crossover"] == -1).sum()
    logger.info(f"           {buys} BUY crossovers · {sells} SELL crossovers found")

    # ── 3. Backtest ───────────────────────────────────────────
    logger.info("[ 3 / 4 ]  Running backtest...")
    result = backtest(df)
    print_backtest_results(result)

    # ── 4. Live signal + paper trade ──────────────────────────
    logger.info("[ 4 / 4 ]  Checking live signal...")
    latest = df.iloc[-1]
    signal = latest["signal"]
    crossover = latest["crossover"]
    latest_price = latest["close"]
    latest_date = df.index[-1].strftime("%Y-%m-%d")

    signal_label = (
        "BULLISH 🟢" if signal == 1 else
        "BEARISH 🔴" if signal == -1 else
        "NEUTRAL  "
    )
    logger.info(f"           Date   : {latest_date}")
    logger.info(f"           Price  : ${latest_price:.2f}")
    logger.info(f"           Signal : {signal_label}")

    # ── NEW: 3-day crossover probability ──────────────────────
    from strategies.probability_estimator import estimate_crossover_probability
    probs = estimate_crossover_probability(df)
    #logger.info(f"           P(BUY next 3d)  : {probs['buy_3d_pct']}%")
    #logger.info(f"           P(SELL next 3d) : {probs['sell_3d_pct']}%")
    logger.info(f"           Basis           : {probs['explanation']}")

    # Only place a paper order if Alpaca keys are configured
    alpaca_ready = ALPACA_API_KEY and ALPACA_API_KEY != "YOUR_KEY"

    if alpaca_ready:
        account = get_account()
        if account:
            cash = float(account.cash)
            logger.info(f"           Alpaca cash: ${cash:,.2f}")
        else:
            cash = float(INITIAL_CAPITAL)

        if crossover == 1:
            # Use full df to compute ATR-based stop
            qty = calculate_position_size(
                balance=cash,
                risk_percent=RISK_PERCENT,
                entry_price=latest_price,
                stop_price=None,  # ← None = auto ATR stop
                max_allocation=MAX_ALLOCATION,
                kelly_fraction=0.5  # can be tuned or passed from backtest
            )
            if qty > 0:
                logger.info(f"           ★ BUY crossover — ordering {qty}x {SYMBOL} "
                            f"(risk-based sizing)")
                buy(SYMBOL, qty)
            else:
                logger.info("           BUY signal but qty=0 (risk/drawdown limit)")

        elif crossover == -1:
            # For sell we usually close full position — but you can add risk logic here later
            logger.info(f"           ★ SELL crossover — ordering {SYMBOL}")
            sell(SYMBOL, 1)  # or query current position from Alpaca

        else:
            logger.info("           No crossover today — holding position")
    else:
        logger.info("           Alpaca keys not set — skipping live order")
        logger.info("           Add keys to .env to enable paper trading")

    logger.info(SEP + "\n")


if __name__ == "__main__":
    run()
