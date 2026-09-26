# QuantBot V1 Changelog

## [V1.0.0] - 2026-09-26

### Fixed
- **MOO3 Logging Contradiction**: Fixed `load_and_register_moo3()` to return `True` after successful model load (was missing return statement), eliminating contradictory log messages
- **Walk-Forward Reporting**: Now explicitly shows traded folds vs zero-trade folds with clear labeling of metrics calculated on traded folds only
- **Drawdown Warnings**: Added explicit warnings for concerning drawdown metrics (< -50%) in backtest output
- **Contribution Analysis**: New reporting shows each strategy's percentage contribution to final decisions

### Added
- **284 Ticker Universe**: Expanded from 10 to 284 tickers across multiple sectors for multi-ticker scanning
- **Contribution Analysis Function**: `log_contribution_analysis()` in strategy_engine.py provides ensemble diversity insights
- **Test Suite**: 
  - `test_imports.py` — Module import validation
  - `test_strategy_engine.py` — Strategy engine functionality
  - `test_backtest.py` — Backtesting and walk-forward validation
- **Enhanced .gitignore**: Properly excludes machine-specific SSL certificates and trained models

### Changed
- **MOO3 Grade Logic**: Now requires minimum 50% trade coverage for PASS grade
- **MC Grade Explanation**: Added clarification that PASS means robustness, not buy-and-hold outperformance
- **Backtest Drawdown Labels**: More granular thresholds (comfortable/moderate/high/very high)

### Technical Details
- All 46 Python files pass syntax validation
- Import tests verify all core modules load correctly
- Strategy engine tests confirm proper signal generation
- Backtest tests validate historical simulation accuracy

---

## Usage

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (optional - runs on simulated data without)
# Create .env file with:
# ALPACA_API_KEY=your_key
# ALPACA_SECRET_KEY=your_secret

# 3. Run the bot
python main.py
```

### Test Suite
```bash
# Run all tests
python test_imports.py          # Module validation
python test_strategy_engine.py  # Strategy engine test
python test_backtest.py         # Backtesting validation
```

### Train MOO3 Model
```bash
# Train a new genetic programming strategy
python quant-trading-bot/genetic/run_genetic.py --pop 50 --gens 50

# Quick test run
python quant-trading-bot/genetic/run_genetic.py --pop 20 --gens 20 --symbol AAPL
```

### Configuration
Edit `quant-trading-bot/config/settings.py` to customize:
- `USE_MULTI_TICKER=True/False` — Enable/disable multi-ticker scanning
- `MIN_SIGNAL_PROBABILITY=60` — Probability threshold for trades
- `USE_MOO3_PLUGIN=True/False` — Enable/disable GP strategy
- `MOO3_PLUGIN_WEIGHT=2.5` — MOO3 contribution weight

### Expected Output
```
STRATEGY ENGINE  =>  BUY (score:+5.30  regime:trending)
CONTRIBUTION ANALYSIS
  Strategy              Weight  Contribution  % of Score
  MACD                    1.5         +0.700        13.2%
  Stochastic              1.0         +1.300        24.5%
  OBV                     1.0         +0.800        15.1%
  MOO3                    2.5         +2.500        47.2%
  MOO3 contribution: 47.2% of total positive score
  ⚠️  MOO3 dominates the ensemble decision - consider reducing weight
```

### V1 Pipeline Flow
```
MARKET DATA
    ↓
Multi-Ticker Scan (284 symbols)
    ↓
Indicator Engine (MACD, RSI, BB, Stoch, ADX, OBV)
    ↓
Strategy Ensemble (6 plugins vote)
    ↓
Contribution Analysis (weights shown)
    ↓
Engine Backtest + Walk-Forward + Monte Carlo
    ↓
Risk Dashboard + Probability Gate
    ↓
Paper Trade (Alpaca) or Signal Rejected
```