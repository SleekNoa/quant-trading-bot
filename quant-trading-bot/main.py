"""
QuantBot — Self-Evolving Multi-Strategy Paper Trading Bot
==========================================================
Run:
    python main.py

Pipeline (7 phases):
    1. Market data          (yfinance / simulated fallback)
    2. Indicator prep       (MACD, RSI, BB, Stoch, ADX, OBV columns)
    3. Regime detection     (trending vs ranging — drives engine weights)
    4. Strategy engine      (6 plugins vote: MACD, RSI, BB, Stoch, SMA, DC)
    5. Engine backtest      (per-strategy accuracy + ensemble metrics)
                            ├─ Walk-Forward Validation  [USE_WALK_FORWARD]
                            └─ Monte Carlo Robustness   [USE_BACKTEST_MC]
    6. Risk evaluation      (VaR sizing, circuit-breakers, MC stress test)
    7. Paper trade          (Alpaca — only fires on high-conviction signals)
    [opt] Multi-ticker scan (ranks signals across TICKERS before Phase 1
                             when USE_MULTI_TICKER = True)

Strategy framework:
    Long, Kampouridis & Papastylianou (2026). Multi-objective GP-based
    algorithmic trading using directional changes. AI Review 59:39.
    -> DC events, regime-aware weights, ensemble voting.

Risk framework:
    Wang, Zhao & Wang (2026). Integrated financial risk management
    framework. Case Studies in Thermal Engineering. ScienceDirect.
    -> VaR/CVaR sizing, circuit-breakers, Monte Carlo stress testing.

Probability gate:
    LogisticProbabilityModel — replaces broken Brownian-motion estimator.
    Trained on 6 engineered features; outputs calibrated P(up) 0-1.
    -> strategies/probability_estimator.py -> models/logistic_probability.py

Walk-forward validation:
    Deep et al. (2025). Walk-Forward Validation Framework. arXiv:2512.12924.
    -> backtesting/walk_forward.py

Monte Carlo robustness:
    Ahmed (2023). Sizing Strategies. arXiv:2309.09094.
    -> backtesting/monte_carlo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# ── Auto-register all strategy plugins ────────────────────────────────────────
import strategies.plugins  # noqa — triggers @register_strategy decorators

from data.market_data import get_historical_data
from strategies.strategy_engine import (
    evaluate_strategies, detect_market_regime,
    log_engine_report, list_strategies,
)
from strategies.adx_filter          import add_adx
from strategies.obv_filter          import add_obv
from strategies.macd_strategy       import generate_signals as macd_gen
from strategies.rsi_strategy        import generate_signals as rsi_gen
from strategies.bollinger_strategy  import generate_signals as bollinger_gen
from strategies.stochastic_strategy import generate_signals as stochastic_gen
from strategies.probability_estimator import (
    estimate_crossover_probability,
    reset_probability_model,
)
from backtesting.backtester import backtest_engine, log_per_strategy_report
from execution.broker import buy, sell, get_account, get_position
from risk.risk_manager import (
    calculate_position_size, compute_historical_var, compute_cvar,
    check_drawdown_circuit_breaker, passes_signal_gate,
    monte_carlo_stress_test, get_stop_loss_price,
)
from utils.logger import logger
from config.settings import (
    SYMBOL, INITIAL_CAPITAL, ALPACA_API_KEY,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD, ADX_THRESHOLD, OBV_MA_PERIOD,
    USE_ADX_FILTER, USE_OBV_FILTER,
    VAR_CONFIDENCE, MAX_RISK_PER_TRADE_PCT,
    MAX_PORTFOLIO_DRAWDOWN_PCT, DAILY_LOSS_LIMIT_PCT,
    MIN_SIGNAL_PROBABILITY, USE_STOP_LOSS, STOP_LOSS_PCT,
    RUN_STRESS_TEST,
    # Walk-forward
    USE_WALK_FORWARD, WF_TRAIN_BARS, WF_TEST_BARS, WF_STEP_BARS,
    # Monte Carlo
    USE_BACKTEST_MC, MC_SIMULATIONS, MC_DD_SIMULATIONS, MC_SEED,
    # Multi-ticker
    USE_MULTI_TICKER, TICKERS, MULTI_TICKER_DELAY_SEC,
    MULTI_TICKER_MIN_BARS, MULTI_TICKER_TOP_N, TICKER_ALLOC_METHOD,
    # Strategy
    STRATEGY,
)

SEP  = "=" * 70
SEP2 = "-" * 70


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _pct(v):  return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"
def _usd(v):  return f"${v:>12,.2f}"


# ── Indicator prep ─────────────────────────────────────────────────────────────

def prepare_indicators(df):
    """
    Add all indicator columns needed by the plugin strategies.
    Pre-computing here avoids redundant rolling calculations on every engine call.
    """
    df = macd_gen(df)       # adds macd, macd_signal, signal, crossover
    df = rsi_gen(df)        # adds rsi
    df = bollinger_gen(df)  # adds bb_upper, bb_lower, bb_mid
    df = stochastic_gen(df) # adds %K, %D
    df = add_adx(df)        # adds adx, adx_trending
    df = add_obv(df)        # adds obv, obv_ma, obv_rising
    return df


# ── Backtest header ────────────────────────────────────────────────────────────

def print_backtest_header(result, symbol=SYMBOL):
    sharpe_note = (
        "  [good]"     if result["sharpe"] > 1.0 else
        "  [marginal]" if result["sharpe"] > 0.5 else
        "  [poor]"
    )
    dd_note = (
        "  [comfortable]" if result["max_drawdown"] > -10 else
        "  [moderate]"    if result["max_drawdown"] > -20 else
        "  [high]"
    )

    logger.info(SEP)
    logger.info(f"  ENGINE BACKTEST  --  {symbol}")
    logger.info(SEP)
    logger.info(f"  {'Initial Capital':<26} {_usd(INITIAL_CAPITAL)}")
    logger.info(f"  {'Final Value':<26} {_usd(result['final_value'])}  <- {_pct(result['return_pct'])}")
    logger.info(SEP2)
    logger.info(f"  {'Buy & Hold':<26} {'':>13}  {_pct(result['buy_hold'])}")
    logger.info(f"  {'Alpha (vs B&H)':<26} {'':>13}  {_pct(result['return_pct'] - result['buy_hold'])}")
    logger.info(SEP2)
    logger.info(f"  {'Sharpe Ratio':<26} {result['sharpe']:>13.2f}{sharpe_note}")
    logger.info(f"  {'Max Drawdown':<26} {result['max_drawdown']:>12.2f}%{dd_note}")
    logger.info(f"  {'Avg Trade Return':<26} {result['avg_trade_pct']:>12.2f}%")
    logger.info(SEP2)
    logger.info(f"  {'Win Rate':<26} {result['win_rate']:>12.1f}%")
    logger.info(f"  {'Total Trades':<26} {result['n_trades']:>13}")
    logger.info(SEP)


# ── Risk dashboard ─────────────────────────────────────────────────────────────

def print_risk_dashboard(cash, equity, var, cvar):
    logger.info(SEP2)
    logger.info("  RISK DASHBOARD")
    logger.info(SEP2)
    logger.info(f"  {'Portfolio Equity':<28} {_usd(equity)}")
    logger.info(f"  {'Available Cash':<28} {_usd(cash)}")
    logger.info(f"  {'1-day VaR (95%)':<28} {var*100:>12.2f}%  (${equity*var:,.0f} at risk)")
    logger.info(f"  {'CVaR (tail)':<28} {cvar*100:>12.2f}%  (${equity*cvar:,.0f} expected)")
    logger.info(f"  {'Max risk/trade':<28} {MAX_RISK_PER_TRADE_PCT*100:>12.1f}%")
    logger.info(f"  {'Drawdown circuit-break':<28} {MAX_PORTFOLIO_DRAWDOWN_PCT*100:>11.0f}%")
    if USE_STOP_LOSS:
        logger.info(f"  {'Stop-loss':<28} {'ON -'+str(int(STOP_LOSS_PCT*100))+'%':>13}")
    else:
        logger.info(f"  {'Stop-loss':<28} {'OFF':>13}")
    logger.info(f"  {'MC stress test':<28} {'ON' if RUN_STRESS_TEST else 'OFF':>13}")
    logger.info(f"  {'Min signal prob':<28} {MIN_SIGNAL_PROBABILITY:>12}%")
    logger.info(SEP2)


# ── Walk-forward runner ────────────────────────────────────────────────────────

def run_walk_forward(df):
    """Run walk-forward validation and print the report."""
    try:
        from backtesting.walk_forward import walk_forward_test, print_walk_forward_report
        from strategies import STRATEGY_FACTORY

        strategy_func = STRATEGY_FACTORY.get(STRATEGY)
        if strategy_func is None:
            logger.warning(f"           Walk-forward: unknown strategy '{STRATEGY}', using MACD")
            strategy_func = STRATEGY_FACTORY.get("macd", list(STRATEGY_FACTORY.values())[0])

        logger.info(
            f"           Walk-forward params: "
            f"train={WF_TRAIN_BARS} bars, test={WF_TEST_BARS} bars, "
            f"step={WF_STEP_BARS or WF_TEST_BARS} bars"
        )

        results, summary = walk_forward_test(
            df,
            strategy_func,
            train_bars=WF_TRAIN_BARS,
            test_bars=WF_TEST_BARS,
            step_bars=WF_STEP_BARS,
        )
        print_walk_forward_report(results, summary)

    except ImportError as exc:
        logger.warning(f"           Walk-forward skipped (import error): {exc}")
    except Exception as exc:
        logger.warning(f"           Walk-forward error: {exc}")


# ── Monte Carlo runner ─────────────────────────────────────────────────────────

def run_backtest_monte_carlo(result):
    """Extract trade returns from backtest result and run MC robustness test."""
    try:
        from backtesting.monte_carlo import (
            monte_carlo_test, monte_carlo_max_drawdown,
            extract_trade_returns, print_monte_carlo_report,
        )

        trades  = result.get("trades", [])
        returns = extract_trade_returns(trades)

        if len(returns) < 3:
            logger.info(
                f"           Monte Carlo skipped: "
                f"need ≥ 3 closed trades, got {len(returns)}"
            )
            return

        logger.info(
            f"           Running {MC_SIMULATIONS:,} MC paths "
            f"over {len(returns)} closed trades..."
        )

        mc = monte_carlo_test(
            returns,
            simulations=MC_SIMULATIONS,
            initial_capital=float(INITIAL_CAPITAL),
            seed=MC_SEED,
        )
        dd = monte_carlo_max_drawdown(
            returns,
            simulations=MC_DD_SIMULATIONS,
            initial_capital=float(INITIAL_CAPITAL),
            seed=MC_SEED,
        )
        print_monte_carlo_report(mc, dd)

    except ImportError as exc:
        logger.warning(f"           Monte Carlo skipped (import error): {exc}")
    except Exception as exc:
        logger.warning(f"           Monte Carlo error: {exc}")


# ── Multi-ticker scanner ───────────────────────────────────────────────────────

def run_multi_ticker_scan() -> str:
    """
    Scan all TICKERS, rank signals, and return the best symbol to trade.

    Returns the ticker with the highest BUY score, or SYMBOL if no BUY
    signals were found.
    """
    try:
        from data.multi_ticker import fetch_all_tickers
        from portfolio.signal_ranker import (
            rank_signals, allocate_capital, print_ranking_report,
        )
        from strategies import STRATEGY_FACTORY
        from backtesting.backtester import backtest_engine as backtest

        logger.info(SEP)
        logger.info(f"  MULTI-TICKER SCAN  ({len(TICKERS)} tickers)")
        logger.info(SEP)

        ticker_data = fetch_all_tickers(
            TICKERS,
            delay_sec=MULTI_TICKER_DELAY_SEC,
            min_bars=MULTI_TICKER_MIN_BARS,
        )

        if not ticker_data:
            logger.warning("           No tickers loaded — falling back to SYMBOL")
            return SYMBOL

        strategy_func = STRATEGY_FACTORY.get(STRATEGY)
        if strategy_func is None:
            strategy_func = STRATEGY_FACTORY.get("macd", list(STRATEGY_FACTORY.values())[0])

        ticker_results: dict = {}

        for ticker, df in ticker_data.items():
            try:
                # Prepare indicators for this ticker
                _df = prepare_indicators(df)

                # Get engine decision
                regime   = detect_market_regime(_df, ADX_THRESHOLD)
                decision, report = evaluate_strategies(_df, regime=regime)

                # Quick backtest for Sharpe
                _signals = strategy_func(_df)
                bt       = backtest(_signals)

                # Probability estimate (reset model per ticker)
                reset_probability_model()
                probs = estimate_crossover_probability(_df)

                ticker_results[ticker] = {
                    "decision":   decision,
                    "sharpe":     bt.get("sharpe",     0.0),
                    "return_pct": bt.get("return_pct", 0.0),
                    "prob_up":    probs["buy_3d_pct"] / 100.0,
                    "price":      float(_df.iloc[-1]["close"]),
                    "n_trades":   bt.get("n_trades",   0),
                }

            except Exception as exc:
                logger.warning(f"           {ticker}: analysis failed — {exc}")
                continue

        # Reset for the main pipeline
        reset_probability_model()

        ranked = rank_signals(ticker_results, top_n=MULTI_TICKER_TOP_N)
        if ranked:
            allocate_capital(ranked, float(INITIAL_CAPITAL), method=TICKER_ALLOC_METHOD)
            # Show which symbol will be used for execution
            best_ticker = ranked[0]["ticker"]
            print_ranking_report(ranked, float(INITIAL_CAPITAL), symbol_in_use=best_ticker)
            logger.info(f"           Best signal: {best_ticker}  (score={ranked[0]['score']:.3f})")
            return best_ticker

        logger.info("           No actionable BUY signals found — using configured SYMBOL")
        return SYMBOL

    except ImportError as exc:
        logger.warning(f"           Multi-ticker scan skipped (import error): {exc}")
        return SYMBOL
    except Exception as exc:
        logger.warning(f"           Multi-ticker scan error: {exc} — using SYMBOL")
        return SYMBOL


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run():
    logger.info(SEP)
    logger.info(f"  QuantBot  |  {SYMBOL}  |  6-plugin engine")
    logger.info(f"  Registered strategies: {', '.join(list_strategies())}")
    logger.info(f"  Prob gate: LogisticModel  |  "
                f"WF: {'ON' if USE_WALK_FORWARD else 'OFF'}  |  "
                f"MC: {'ON' if USE_BACKTEST_MC else 'OFF'}  |  "
                f"Multi-ticker: {'ON' if USE_MULTI_TICKER else 'OFF'}")
    logger.info(SEP)

    # ── [opt] Multi-ticker pre-scan ────────────────────────────────────────────
    # Runs before Phase 1 when USE_MULTI_TICKER is True.
    # Scans TICKERS, ranks signals, returns the best symbol for the pipeline.
    active_symbol = SYMBOL
    if USE_MULTI_TICKER:
        logger.info("[M-T]  Multi-ticker scan running before Phase 1...")
        active_symbol = run_multi_ticker_scan()
        logger.info(f"[M-T]  Pipeline will execute on: {active_symbol}")

    # ── 1. Market Data ─────────────────────────────────────────────────────────
    logger.info("[ 1 / 7 ]  Downloading market data...")
    df = get_historical_data(symbol=active_symbol) if USE_MULTI_TICKER else get_historical_data()
    logger.info(f"           {len(df)} bars loaded ({df.index[0].date()} -> {df.index[-1].date()})")

    # ── 2. Compute all indicator columns ──────────────────────────────────────
    logger.info("[ 2 / 7 ]  Computing indicators...")
    df = prepare_indicators(df)
    logger.info("           MACD / RSI / Bollinger / Stochastic / ADX / OBV ready")

    # ── 3. Regime detection ───────────────────────────────────────────────────
    logger.info("[ 3 / 7 ]  Detecting market regime...")
    regime = detect_market_regime(df, ADX_THRESHOLD)

    # ── 4. Strategy engine + probability gate (live signal) ───────────────────
    logger.info("[ 4 / 7 ]  Running strategy engine (live signal)...")
    decision, report = evaluate_strategies(df, regime=regime)
    log_engine_report(report)

    latest      = df.iloc[-1]
    price       = float(latest["close"])
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    logger.info(f"           Date  : {latest_date}   Price : ${price:.2f}")

    # Probability gate — LogisticProbabilityModel (replaces Brownian estimator)
    probs = estimate_crossover_probability(df)
    logger.info(f"           Forecast: {probs['explanation']}")

    # ── 5. Engine backtest ────────────────────────────────────────────────────
    logger.info("[ 5 / 7 ]  Running engine backtest...")
    result = backtest_engine(df)
    print_backtest_header(result, symbol=active_symbol)
    if "per_strategy" in result:
        log_per_strategy_report(result["per_strategy"])

    # ── 5a. Walk-Forward Validation ───────────────────────────────────────────
    if USE_WALK_FORWARD:
        logger.info("[ 5a ]  Walk-forward validation (out-of-sample)...")
        run_walk_forward(df)
    else:
        logger.info("[ 5a ]  Walk-forward: disabled (USE_WALK_FORWARD=False)")

    # ── 5b. Monte Carlo Robustness Test ───────────────────────────────────────
    if USE_BACKTEST_MC:
        logger.info("[ 5b ]  Monte Carlo backtest robustness...")
        run_backtest_monte_carlo(result)
    else:
        logger.info("[ 5b ]  Monte Carlo backtest: disabled (USE_BACKTEST_MC=False)")

    # ── 6+7. Risk evaluation + paper trade ────────────────────────────────────
    logger.info("[ 6 / 7 ]  Risk evaluation + paper trade...")

    alpaca_ready = bool(ALPACA_API_KEY and ALPACA_API_KEY not in ("", "YOUR_KEY"))
    if not alpaca_ready:
        logger.info("           Alpaca keys not configured — skipping live orders")
        logger.info(SEP + "\n")
        return

    account = get_account()
    if account is None:
        logger.error("           Cannot reach Alpaca — aborting")
        logger.info(SEP + "\n")
        return

    cash   = account["cash"]
    equity = account["equity"]
    var    = compute_historical_var(df)
    cvar   = compute_cvar(df)
    print_risk_dashboard(cash, equity, var, cvar)

    # Drawdown circuit-breaker
    peak = max(equity, float(INITIAL_CAPITAL))
    if check_drawdown_circuit_breaker(equity, peak):
        logger.info(SEP + "\n")
        return

    # No actionable signal
    if decision == "HOLD":
        position = get_position(active_symbol)
        if position:
            logger.info(
                f"           Holding {position['qty']}x {active_symbol}  "
                f"(P&L: ${position['unrealized_pl']:+,.2f})"
            )
        else:
            logger.info("           Engine says HOLD — flat, no action")
        logger.info(SEP + "\n")
        return

    # Signal quality gate (probability filter from LogisticModel)
    cross_dir = 1 if decision == "BUY" else -1
    if not passes_signal_gate(probs["buy_3d_pct"], probs["sell_3d_pct"], cross_dir):
        logger.info("           Trade blocked by signal quality gate (logistic probability)")
        logger.info(SEP + "\n")
        return

    if decision == "SELL":
        logger.info("[ 7 / 7 ]  Executing SELL...")
        sell(active_symbol)
        logger.info(SEP + "\n")
        return

    # BUY path
    logger.info("[ 7 / 7 ]  Executing BUY...")
    qty = calculate_position_size(cash, price, df)
    if qty <= 0:
        logger.warning("           qty=0 — insufficient capital or risk budget exceeded")
        logger.info(SEP + "\n")
        return

    stress = monte_carlo_stress_test(df, price, qty)
    if not stress["passed"]:
        logger.info("           Trade blocked by Monte Carlo stress test (live gate)")
        logger.info(SEP + "\n")
        return

    stop_price = get_stop_loss_price(price) if USE_STOP_LOSS else None
    buy(active_symbol, qty, stop_loss_price=stop_price)
    logger.info(SEP + "\n")


if __name__ == "__main__":
    run()