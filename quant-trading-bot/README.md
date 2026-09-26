# QuantBot — Self-Evolving Multi-Strategy Paper Trading Bot

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add API keys to `.env`
```
ALPHAVANTAGE_API_KEY=your_key   # alphavantage.co — free, no card
ALPACA_API_KEY=your_key         # app.alpaca.markets → Paper Trading
ALPACA_SECRET_KEY=your_secret
```

### 3. Run (Single Ticker Mode)
```bash
python main.py
```

### 4. Run (Multi-Ticker Scan Mode)
The bot scans 284 tickers and executes on the best signal:
```bash
# Multi-ticker is enabled by default in config/settings.py
python main.py
```

No keys? It runs on simulated data automatically.

---

## V1 Features

### Core Pipeline (7 Phases)
1. **Market Data** — yfinance / Alpha Vantage with SSL certificate fix
2. **Indicator Prep** — MACD, RSI, BB, Stochastic, ADX, OBV
3. **Regime Detection** — Trending vs ranging via ADX
4. **Strategy Engine** — 6-plugin ensemble with voting
5. **Engine Backtest** — Historical validation + walk-forward + Monte Carlo
6. **Risk Evaluation** — VaR/CVaR sizing, circuit breakers, stress testing
7. **Paper Trade** — Alpaca execution (only on high-conviction signals)

### Strategy Engine
- **MACD** — Momentum crossover with regime weighting
- **RSI** — Mean reversion oscillator
- **Bollinger** — Volatility breakout
- **Stochastic** — Overbought/oversold timing
- **SMA** — Trend following
- **Directional Change (DC)** — Event-driven signals
- **MOO3** — Genetic programming evolved strategy (optional)

### Risk Management
- **Probability Gate** — Logistic model requires P(up) ≥ 60%
- **VaR Position Sizing** — Risk-adjusted position calculation
- **Monte Carlo Stress Test** — 10,000-path robustness check
- **Circuit Breakers** — Drawdown and daily loss limits
- **Exit Monitor** — Trailing stop, time-based, take-profit

### Validation
- **Walk-Forward** — Rolling out-of-sample validation (17 folds)
- **Monte Carlo** — Permutation-based sequence risk analysis
- **Contribution Analysis** — Shows each strategy's impact on decisions

---

## Project Structure

```
quant-trading-bot/
├── main.py                                 ← Entry point
├── fix_ssl_corporate.py                    ← SSL certificate fix (machine-specific)
├── TICKERS.csv                             ← 284-ticker universe
│
├── quant-trading-bot/
│   ├── config/settings.py                  ← All configuration parameters
│   ├── data/market_data.py                 ← Yahoo/Alpha Vantage fetch
│   ├── data/multi_ticker.py                ← Multi-ticker scanner
│   │
│   ├── strategies/
│   │   ├── strategy_engine.py              ← Multi-strategy voting engine
│   │   ├── plugins.py                      ← Strategy registration
│   │   ├── macd_strategy.py                ← MACD signals
│   │   ├── rsi_strategy.py                 ← RSI signals
│   │   ├── bollinger_strategy.py           ← BB signals
│   │   ├── stochastic_strategy.py          ← Stochastic signals
│   │   ├── moving_average_strategy.py      ← SMA signals
│   │   ├── directional_change_strategy.py  ← DC signals
│   │   ├── adx_filter.py                   ← Trend strength filter
│   │   ├── obv_filter.py                   ← Volume confirmation
│   │   └── probability_estimator.py        ← P(up) model
│   │
│   ├── backtesting/
│   │   ├── backtester.py                   ← Historical simulation
│   │   ├── walk_forward.py                 ← WF validation
│   │   └── monte_carlo.py                  ← MC robustness test
│   │
│   ├── genetic/
│   │   ├── gp_engine.py                    ← MOO3 GP engine
│   │   ├── run_genetic.py                  ← MOO3 CLI trainer
│   │   ├── fitness.py                      ← Trade simulation
│   │   ├── gp_tree.py                      ← Tree structures
│   │   ├── nsga2.py                      ← NSGA-II selection
│   │   └── sharpe_selector.py              ← Modified Sharpe
│   │
│   ├── risk/risk_manager.py                ← VaR/CVaR, sizing
│   ├── execution/broker.py                 ← Alpaca paper trading
│   ├── execution/exit_monitor.py           ← Position exits
│   ├── portfolio/signal_ranker.py          ← Multi-ticker ranking
│   └── utils/logger.py                     ← Logging utilities
│
├── models/                                 ← Trained models (gitignored)
│   └── moo3_best.pkl
│
├── logs/                                   ← Log files (gitignored)
├── .env                                    ← API keys (never commit)
├── requirements.txt
└── Dockerfile
```

---

## Testing

### Run Validation Tests
```bash
python test_imports.py          # Verify all modules import
python test_strategy_engine.py  # Test strategy evaluation
python test_backtest.py         # Test backtesting engine
```

### Train MOO3 Model
```bash
python quant-trading-bot/genetic/run_genetic.py --pop 50 --gens 50
```

---

## Configuration (`config/settings.py`)

| Setting | Default | Description |
|---|---|---|
| `USE_MULTI_TICKER` | True | Scan 284 tickers for best signal |
| `USE_MOO3_PLUGIN` | True | Use trained GP strategy |
| `MIN_SIGNAL_PROBABILITY` | 60 | P(up) threshold for trades |
| `INITIAL_CAPITAL` | 100,000 | Starting paper-trade capital |
| `STOP_LOSS_PCT` | 5% | Per-trade stop loss |
| `TAKE_PROFIT_PCT` | 8% | Per-trade take profit |
| `MOO3_PLUGIN_WEIGHT` | 2.5 | MOO3 contribution weight |

---

## V1 Release Notes

### Fixed Issues
- ✅ MOO3 logging contradiction resolved
- ✅ Walk-forward reporting now distinguishes traded vs zero-trade folds
- ✅ Backtest warnings for concerning drawdown metrics
- ✅ Contribution analysis shows strategy impact percentages

### Known Considerations
- SSL certificate fix (`fix_ssl_corporate.py`) is machine-specific
- Model files (`.pkl`) in `models/` are gitignored
- `.env` file with API keys is gitignored

---

## Workflow

**Phase 1 — Backtest** · Test on historical data with walk-forward validation  
**Phase 2 — Paper trade** · Run via Alpaca paper account for 2–4 weeks  
**Phase 3 — Live trade** · Start with small position, scale gradually