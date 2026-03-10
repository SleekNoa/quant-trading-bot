# QuantBot — Moving Average Crossover Trading Bot

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
SYMBOL=AAPL
```

### 3. Run
```bash
python main.py
```

No keys? It runs on simulated data automatically.

---

## Project Structure

```
quant-trading-bot/
├── config/settings.py                    ← MA windows, capital, risk %
├── data/market_data.py                   ← Alpha Vantage fetch + sim fallback
├── strategies/moving_average_strategy.py ← BUY/SELL signal generation
├── risk/risk_manager.py                  ← Position sizing
├── execution/broker.py                   ← Alpaca paper order execution
├── backtesting/backtester.py             ← Historical P&L simulation
├── utils/logger.py                       ← Logging setup
├── main.py                               ← Entry point
├── .env                                  ← API keys (never commit this)
├── requirements.txt
└── Dockerfile
```

---

## Docker

```bash
docker build -t trading-bot .
docker run --env-file .env trading-bot
```

---

## Configuration (`config/settings.py`)

| Setting | Default | Description |
|---|---|---|
| `SHORT_WINDOW` | 50 | Short MA period |
| `LONG_WINDOW` | 200 | Long MA period |
| `INITIAL_CAPITAL` | 10000 | Starting capital ($) |
| `RISK_PERCENT` | 0.01 | Risk per trade (1%) |

---

## Testing Workflow

**Phase 1 — Backtest** · Run on AAPL, SPY, QQQ, MSFT, TSLA  
**Phase 2 — Paper trade** · Run for 2–4 weeks via Alpaca paper account  
**Phase 3 — Live trade** · Start with $100, scale gradually
