"""
genetic/run_genetic.py — MOO3 CLI Entry Point
===============================================
Standalone script to train and save the MOO3 GP strategy.

Usage
-----
    # Train on default symbol with default settings
    python genetic/run_genetic.py

    # Smaller/faster run for testing
    python genetic/run_genetic.py --pop 20 --gens 20 --symbol AAPL

    # Full paper-spec run with parallelism
    python genetic/run_genetic.py --pop 50 --gens 50 --workers 4

    # Load saved model and print its signal on today's data
    python genetic/run_genetic.py --predict-only

After training the model is saved to models/moo3_best.pkl and
auto-registered on the next run of main.py.

Integration with main.py
------------------------
    Add to the top of main.py (after plugin imports):

        from genetic.gp_engine import load_and_register_moo3
        load_and_register_moo3(df)   # registers "MOO3" plugin if model exists

Pipeline position
-----------------
    Typically trained offline (this script) then used live (main.py).
    Re-train periodically to adapt to changing market conditions.
    Long et al. (2026): "It may be necessary to periodically re-train the
    algorithm to capture new changes in market conditions."
"""

import sys
import os
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from genetic.gp_engine import MOO3Engine, MOO3Individual
from data.market_data import get_historical_data
from config.settings import SYMBOL, INITIAL_CAPITAL
from utils.logger import logger

# Import and trigger all indicator computations
from strategies.macd_strategy       import generate_signals as macd_gen
from strategies.rsi_strategy        import generate_signals as rsi_gen
from strategies.bollinger_strategy  import generate_signals as bollinger_gen
from strategies.stochastic_strategy import generate_signals as stochastic_gen
from strategies.adx_filter          import add_adx
from strategies.obv_filter          import add_obv

try:
    from strategies.directional_change_strategy import add_dc_indicators
    _HAS_DC = True
except ImportError:
    _HAS_DC = False


SEP  = "=" * 70
SEP2 = "-" * 70


# ── Indicator preparation ──────────────────────────────────────────────────────

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicator computations to the raw OHLCV dataframe."""
    df = macd_gen(df)
    df = rsi_gen(df)
    df = bollinger_gen(df)
    df = stochastic_gen(df)
    df = add_adx(df)
    df = add_obv(df)
    if _HAS_DC:
        try:
            df = add_dc_indicators(df, theta=0.01)
        except Exception as e:
            logger.warning(f"DC indicators failed: {e} — using TA-only terminal set")
    # Rename stochastic columns to expected names if needed
    if "%K" in df.columns and "stoch_k" not in df.columns:
        df = df.rename(columns={"%K": "stoch_k", "%D": "stoch_d"})
    return df


# ── Training pipeline ──────────────────────────────────────────────────────────

def train(
    symbol:   str = SYMBOL,
    pop_size: int = 50,
    n_gens:   int = 50,
    workers:  int = 1,
    verbose:  bool = True,
) -> MOO3Individual:
    """Download data, prepare indicators, train MOO3, save model."""

    print(SEP)
    print(f"  MOO3 Training Run")
    print(f"  Symbol: {symbol}  |  Pop: {pop_size}  |  Gens: {n_gens}")
    print(SEP)

    # ── 1. Download data ──────────────────────────────────────────────────────
    print("  [1/4] Downloading market data...")
    df = get_historical_data(symbol)
    if df is None or len(df) < 100:
        raise RuntimeError(f"Insufficient data for {symbol} (got {len(df) if df is not None else 0} bars)")
    print(f"        {len(df)} bars loaded  ({df.index[0].date()} → {df.index[-1].date()})")

    # ── 2. Compute indicators ─────────────────────────────────────────────────
    print("  [2/4] Computing indicators...")
    df = prepare_df(df)
    # Drop NaN rows from indicator warm-up
    df = df.dropna(subset=["rsi", "macd"]).copy()
    print(f"        {len(df)} bars after indicator warm-up")

    # ── 3. Train MOO3 ─────────────────────────────────────────────────────────
    print("  [3/4] Running MOO3 GP...")
    engine = MOO3Engine(
        df=df,
        pop_size=pop_size,
        n_gens=n_gens,
        tournament_k=3,
        p_crossover=0.80,
        p_mutation=0.10,
        max_depth=5,
        msr_weights=(0.40, 0.30, 0.30),   # TR, WR, MaxDD
        n_workers=workers,
        verbose=verbose,
    )
    best = engine.run()

    # ── 4. Save model ─────────────────────────────────────────────────────────
    print("  [4/4] Saving model...")
    engine.save()

    return best


# ── Predict-only mode ──────────────────────────────────────────────────────────

def predict_only(symbol: str = SYMBOL) -> None:
    """Load saved model and show its signal on the latest bar."""
    from genetic.gp_engine import MOO3Engine

    print(SEP)
    print("  MOO3 Predict Mode")
    print(SEP)

    df  = get_historical_data(symbol)
    df  = prepare_df(df)
    df  = df.dropna(subset=["rsi", "macd"]).copy()

    try:
        best   = MOO3Engine.load()
        signal = best.predict(df)
        print(f"  Symbol: {symbol}")
        print(f"  Latest bar: {df.index[-1].date()}")
        print(f"  MOO3 Signal: {signal}")
        print(f"  Model: {best}")
    except FileNotFoundError:
        print("  No saved model found. Run without --predict-only first.")
    print(SEP)


# ── Generation progress plot (optional — requires matplotlib) ─────────────────

def plot_history(history: list) -> None:
    try:
        import matplotlib.pyplot as plt

        gens    = [h["gen"] for h in history]
        tr_best = [h["tr_best"] for h in history]
        tr_mean = [h["tr_mean"] for h in history]

        plt.figure(figsize=(10, 4))
        plt.plot(gens, tr_best, label="Best TR", linewidth=2)
        plt.plot(gens, tr_mean, label="Mean TR", linewidth=1, linestyle="--")
        plt.axhline(0, color="grey", linewidth=0.5)
        plt.xlabel("Generation")
        plt.ylabel("Total Return")
        plt.title("MOO3 GP — Total Return Convergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig("models/moo3_convergence.png", dpi=150)
        print("  Convergence plot saved → models/moo3_convergence.png")
    except ImportError:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MOO3 Genetic Programming Trader")
    parser.add_argument("--symbol",       default=SYMBOL,   help="Ticker symbol")
    parser.add_argument("--pop",   type=int, default=50,    help="Population size")
    parser.add_argument("--gens",  type=int, default=50,    help="Number of generations")
    parser.add_argument("--workers", type=int, default=1,   help="Parallel workers")
    parser.add_argument("--predict-only", action="store_true",
                        help="Only generate signal from saved model (no training)")
    parser.add_argument("--quiet", action="store_true",     help="Suppress per-gen output")
    args = parser.parse_args()

    if args.predict_only:
        predict_only(args.symbol)
        return

    best = train(
        symbol=args.symbol,
        pop_size=args.pop,
        n_gens=args.gens,
        workers=args.workers,
        verbose=not args.quiet,
    )

    print()
    print("  Training complete.")
    print(f"  Best strategy: {best}")
    print(f"  Tree: {best.tree}")
    print()
    print("  To use in main.py, add to the top:")
    print("    from genetic.gp_engine import load_and_register_moo3")
    print("    load_and_register_moo3(df)   # call after df is ready")


if __name__ == "__main__":
    main()