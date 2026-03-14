"""
genetic/gp_engine.py — MOO3 Genetic Programming Engine
=======================================================
Main evolutionary loop implementing Long et al. (2026) MOO3.

Architecture
------------
  MOO3Individual  — one candidate strategy (tree + sell params + objectives)
  MOO3Engine      — orchestrates: initialise → evolve → select → register

Algorithm (paper Section 4, Algorithm 1 + NSGA-II flowchart Fig. 3)
---------------------------------------------------------------------
  1. Initialise population P using ramped half-and-half
  2. Evaluate fitness (TR, WR, MaxDD) for each individual
  3. Repeat for N generations:
       a. Assign Pareto ranks + crowding distances (NSGA-II)
       b. Build mating pool via tournament selection
       c. Apply subtree crossover (prob p_cx) and point mutation (prob p_mut)
       d. Evaluate offspring fitness
       e. Combine parents + offspring (2P individuals)
       f. Select next generation of P via NSGA-II élite strategy
  4. Extract Pareto front (rank-1 individuals)
  5. Select best individual using modified Sharpe Ratio
  6. Register best as "MOO3" plugin in strategy_engine
  7. Pickle best individual to models/moo3_best.pkl

Parameters (defaults from Long et al. 2026 Table 3)
----------------------------------------------------
  pop_size    = 50     (P)
  n_gens      = 50     (N)
  tournament  = 3      (k)
  p_crossover = 0.80
  p_mutation  = 0.10
  max_depth   = 5
"""

from __future__ import annotations

import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import logger

from genetic.gp_tree import (
    Node,
    ramped_half_and_half,
    subtree_crossover,
    point_mutation,
)
from genetic.terminals import TERMINAL_NAMES, build_terminal_matrix
from genetic.fitness import (
    evaluate_individual_with_trades,
    DEFAULT_SELL_DAYS,
    DEFAULT_SELL_PCT,
    DEFAULT_SL_PCT,
    NO_TRADE_PENALTY,
)
from genetic.nsga2 import assign_ranks_and_distances, tournament_select, select_next_generation
from genetic.sharpe_selector import select_from_pareto, describe_pareto_front
from genetic.nsga2 import fast_non_dominated_sort


# ── Model persistence path ────────────────────────────────────────────────────

_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "moo3_best.pkl")


# ── Individual ────────────────────────────────────────────────────────────────

@dataclass
class MOO3Individual:
    """
    One candidate trading strategy.

    tree       : GP boolean expression tree (the evolved buy-signal rule)
    sell_days  : max holding period in bars (evolved alongside tree)
    sell_pct   : take-profit threshold (evolved alongside tree)
    sl_pct     : stop-loss threshold
    objectives : (TR, WR, MaxDD) — set after fitness evaluation
    """
    tree:       Node
    sell_days:  int   = DEFAULT_SELL_DAYS
    sell_pct:   float = DEFAULT_SELL_PCT
    sl_pct:     float = DEFAULT_SL_PCT
    objectives: Tuple[float, float, float] = field(default=NO_TRADE_PENALTY)
    trade_count: int = 0

    # ── Signal generation ──────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> str:
        """
        Generate BUY / HOLD signal for the latest bar.

        Parameters
        ----------
        df : DataFrame with all indicator columns (same as training data)

        Returns "BUY" | "HOLD"
        """
        from genetic.terminals import build_terminal_matrix, row_to_dict
        try:
            matrix, _ = build_terminal_matrix(df)
            row = row_to_dict(matrix, -1)
            if self.tree.evaluate(row):
                return "BUY"
        except Exception:
            pass
        return "HOLD"

    # ── Sell parameters ────────────────────────────────────────────────────────

    def mutate_sell_params(self) -> None:
        """Randomly perturb sell_days and sell_pct (treat as additional genome)."""
        if random.random() < 0.3:
            self.sell_days = max(3, int(self.sell_days + random.gauss(0, 3)))
            self.sell_days = min(self.sell_days, 40)
        if random.random() < 0.3:
            self.sell_pct = float(np.clip(
                self.sell_pct + random.gauss(0, 0.01), 0.02, 0.20
            ))

    def __repr__(self) -> str:
        tr, wr, dd = self.objectives
        return (
            f"MOO3Individual(TR={tr:+.3f}, WR={wr:.1%}, MaxDD={dd:.1%}, "
            f"sell_days={self.sell_days}, sell_pct={self.sell_pct:.1%}, "
            f"trades={self.trade_count})"
        )


# ── MOO3 Engine ───────────────────────────────────────────────────────────────

class MOO3Engine:
    """
    Full MOO3 Genetic Programming engine.

    Usage
    -----
        engine = MOO3Engine(df, pop_size=50, n_gens=50)
        best   = engine.run()          # returns MOO3Individual
        engine.register_as_plugin()    # adds "MOO3" to strategy_engine
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        pop_size:    int   = 50,
        n_gens:      int   = 50,
        tournament_k: int  = 3,
        p_crossover: float = 0.80,
        p_mutation:  float = 0.10,
        max_depth:   int   = 5,
        msr_weights: Tuple[float, float, float] = (0.40, 0.30, 0.30),
        n_workers:   int   = 1,            # >1 enables multiprocessing
        verbose:     bool  = True,
    ) -> None:
        self.df          = df.copy()
        self.pop_size    = pop_size
        self.n_gens      = n_gens
        self.tournament_k = tournament_k
        self.p_crossover = p_crossover
        self.p_mutation  = p_mutation
        self.max_depth   = max_depth
        self.msr_weights = msr_weights
        self.n_workers   = n_workers
        self.verbose     = verbose

        # Pre-build terminal matrix once (expensive to rebuild per eval)
        self.close_prices = df["close"].values.astype(np.float64)
        self.term_matrix, self._norms = build_terminal_matrix(df)

        # Population + tracking
        self.population:  List[MOO3Individual] = []
        self.objectives:  np.ndarray = np.empty((0, 3))   # (pop, 3)
        self.gen_history: List[dict] = []
        self.best_individual: Optional[MOO3Individual] = None

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_population(self) -> None:
        trees = ramped_half_and_half(TERMINAL_NAMES, self.pop_size, self.max_depth)
        self.population = []
        for tree in trees:
            ind = MOO3Individual(
                tree=tree,
                sell_days=random.randint(5, 30),
                sell_pct=random.uniform(0.02, 0.15),
                sl_pct=random.uniform(0.02, 0.40),
            )
            self.population.append(ind)

    # ── Fitness evaluation ─────────────────────────────────────────────────────

    def _evaluate_all(self, individuals: List[MOO3Individual]) -> np.ndarray:
        """
        Evaluate objectives for all individuals.
        Returns (n, 3) array of (TR, WR, MaxDD).
        """
        objectives = np.zeros((len(individuals), 3), dtype=np.float64)
        trade_counts = np.zeros(len(individuals), dtype=np.int64)

        if self.n_workers > 1:
            # Parallel evaluation via ProcessPoolExecutor
            # Note: trees must be picklable (they are pure Python objects)
            from genetic.fitness import evaluate_individual_with_trades as _eval
            futures = {}
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                for i, ind in enumerate(individuals):
                    fut = executor.submit(
                        _eval,
                        ind.tree,
                        self.close_prices,
                        self.term_matrix,
                        ind.sell_days,
                        ind.sell_pct,
                        ind.sl_pct,
                    )
                    futures[fut] = i
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        obj, tcount = fut.result()
                        objectives[i] = obj
                        trade_counts[i] = tcount
                    except Exception:
                        objectives[i] = NO_TRADE_PENALTY
                        trade_counts[i] = 0
        else:
            # Single-threaded (default — avoids process spawn overhead for small P)
            for i, ind in enumerate(individuals):
                try:
                    obj, tcount = evaluate_individual_with_trades(
                        ind.tree,
                        self.close_prices,
                        self.term_matrix,
                        ind.sell_days,
                        ind.sell_pct,
                        ind.sl_pct,
                    )
                    objectives[i] = obj
                    trade_counts[i] = tcount
                except Exception:
                    objectives[i] = NO_TRADE_PENALTY
                    trade_counts[i] = 0

        # Update individual's cached objectives
        for i, ind in enumerate(individuals):
            ind.objectives = tuple(objectives[i])
            ind.trade_count = int(trade_counts[i])

        return objectives

    # ── Offspring generation ───────────────────────────────────────────────────

    def _make_offspring(
        self,
        ranks:     np.ndarray,
        distances: np.ndarray,
    ) -> List[MOO3Individual]:
        """
        Produce pop_size offspring via tournament selection + crossover/mutation.
        """
        offspring: List[MOO3Individual] = []

        while len(offspring) < self.pop_size:
            # Tournament-select two parents
            p1_idx = tournament_select(ranks, distances, self.tournament_k)
            p2_idx = tournament_select(ranks, distances, self.tournament_k)
            p1     = self.population[p1_idx]
            p2     = self.population[p2_idx]

            # Clone trees for genetic operations
            t1 = p1.tree.clone()
            t2 = p2.tree.clone()

            # Subtree crossover
            if random.random() < self.p_crossover:
                t1, t2 = subtree_crossover(t1, t2, max_result_depth=self.max_depth + 2)

            # Point mutation (applied independently to each offspring)
            if random.random() < self.p_mutation:
                t1 = point_mutation(t1, TERMINAL_NAMES)
            if random.random() < self.p_mutation:
                t2 = point_mutation(t2, TERMINAL_NAMES)

            # Build offspring individuals with inherited sell params + noise
            for t, p in [(t1, p1), (t2, p2)]:
                child = MOO3Individual(
                    tree=t,
                    sell_days=p.sell_days,
                    sell_pct=p.sell_pct,
                    sl_pct=p.sl_pct,
                )
                child.mutate_sell_params()
                offspring.append(child)
                if len(offspring) >= self.pop_size:
                    break

        return offspring[:self.pop_size]

    # ── Stats logging ──────────────────────────────────────────────────────────

    def _log_gen(self, gen: int, objectives: np.ndarray, elapsed: float) -> None:
        if not self.verbose:
            return
        tr_mean  = objectives[:, 0].mean()
        tr_best  = objectives[:, 0].max()
        wr_mean  = objectives[:, 1].mean()
        dd_mean  = objectives[:, 2].mean()
        n_front1 = sum(1 for obj in objectives if
                       all(obj[0] >= o[0] and obj[1] >= o[1] and obj[2] <= o[2]
                           for o in objectives))
        print(
            f"  Gen {gen:>3}/{self.n_gens}  "
            f"TR_best={tr_best:+.3f}  TR_mean={tr_mean:+.3f}  "
            f"WR={wr_mean:.1%}  DD={dd_mean:.1%}  "
            f"t={elapsed:.1f}s"
        )
        self.gen_history.append({
            "gen": gen, "tr_best": tr_best, "tr_mean": tr_mean,
            "wr_mean": wr_mean, "dd_mean": dd_mean,
        })

    # ── Main evolutionary loop ─────────────────────────────────────────────────

    def run(self) -> MOO3Individual:
        """
        Execute the full MOO3 algorithm.

        Returns
        -------
        best : MOO3Individual — strategy selected from final Pareto front
               using the modified Sharpe Ratio criterion
        """
        SEP = "=" * 70

        if self.verbose:
            print(SEP)
            print("  MOO3 Genetic Programming Engine")
            print("  Long, Kampouridis & Papastylianou (2026)")
            print(f"  Population={self.pop_size}  Generations={self.n_gens}")
            print(f"  Objectives: Total Return (×0.40)  |  Win Rate (×0.30)  "
                  f"|  MaxDD (×0.30)")
            print(f"  Terminals: {len(TERMINAL_NAMES)} ({len([t for t in TERMINAL_NAMES if t.startswith('dc_')])} DC + TA)")
            print(SEP)

        # ── Step 1: Initialise ────────────────────────────────────────────────
        self._init_population()

        # ── Step 2: Initial fitness evaluation ───────────────────────────────
        t_start = time.perf_counter()
        self.objectives = self._evaluate_all(self.population)
        if self.verbose:
            print(f"  Initial population evaluated in {time.perf_counter()-t_start:.1f}s")

        # ── Step 3: Evolutionary loop ─────────────────────────────────────────
        for gen in range(1, self.n_gens + 1):
            t0 = time.perf_counter()

            # Rank + crowding
            ranks, distances = assign_ranks_and_distances(self.objectives)

            # Generate offspring
            offspring = self._make_offspring(ranks, distances)

            # Evaluate offspring
            offspring_obj = self._evaluate_all(offspring)

            # Combine parents + offspring
            combined_pop = self.population + offspring
            combined_obj = np.vstack([self.objectives, offspring_obj])

            # NSGA-II élite selection
            selected_idx = select_next_generation(combined_obj, self.pop_size)

            self.population = [combined_pop[i] for i in selected_idx]
            self.objectives = combined_obj[selected_idx]

            self._log_gen(gen, self.objectives, time.perf_counter() - t0)

        # ── Step 4: Extract Pareto front ──────────────────────────────────────
        fronts = fast_non_dominated_sort(self.objectives)
        pareto_indices = fronts[0]

        pareto_objectives = self.objectives[pareto_indices]
        pareto_population = [self.population[i] for i in pareto_indices]
        pareto_trade_counts = np.array([ind.trade_count for ind in pareto_population], dtype=np.int64)

        # ── Step 5: Select best via mSR ───────────────────────────────────────
        best_idx    = select_from_pareto(
            pareto_objectives,
            self.msr_weights,
            trade_counts=pareto_trade_counts,
            min_trades=25,
        )
        best        = pareto_population[best_idx]
        self.best_individual = best

        if self.verbose:
            print(SEP)
            print(f"  PARETO FRONT  ({len(pareto_indices)} solutions)")
            print(
                describe_pareto_front(
                    pareto_objectives,
                    self.msr_weights,
                    trade_counts=pareto_trade_counts,
                    min_trades=25,
                )
            )
            print()
            print(f"  SELECTED STRATEGY: {best}")
            print(SEP)

        return best

    # ── Plugin registration ────────────────────────────────────────────────────

    def register_as_plugin(self, weight: float = 2.0) -> None:
        """
        Register the best individual as "MOO3" in the strategy engine.

        The plugin evaluates the evolved GP tree on the latest bar
        and returns BUY / HOLD.  SELL is handled by the existing
        position management logic in main.py.
        """
        if self.best_individual is None:
            raise RuntimeError("No best individual — run MOO3Engine.run() first.")

        best = self.best_individual

        # Import here to avoid circular imports
        from strategies.strategy_engine import register_strategy

        @register_strategy("MOO3", weight=weight)
        def moo3_plugin(df: pd.DataFrame) -> str:
            return best.predict(df)

        if self.verbose:
            print("  [MOO3] Plugin registered in strategy engine  (weight=%.1f)" % weight)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = _MODEL_PATH) -> None:
        """Pickle the best individual to disk."""
        if self.best_individual is None:
            raise RuntimeError("No model to save — run first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.best_individual, f)
        if self.verbose:
            print(f"  [MOO3] Model saved → {path}")

    @staticmethod
    def load(path: str = _MODEL_PATH) -> MOO3Individual:
        """Load a previously trained individual from disk."""
        with open(path, "rb") as f:
            ind = pickle.load(f)
        return ind

    # ── Sanitize MOO3params ─────────────────────────────
    @staticmethod
    def sanitize_moo3_params(ind: "MOO3Individual") -> "MOO3Individual":
        if ind.sl_pct <= 0:
            logger.warning(f"[MOO3] Invalid SL {ind.sl_pct:+.4f} -> set to 0.02")
            ind.sl_pct = 0.02
        if ind.sl_pct > 0.40:
            logger.warning(f"[MOO3] SL too large {ind.sl_pct:+.4f} -> clamped to 0.40")
            ind.sl_pct = 0.40
        if ind.sell_pct <= 0:
            logger.warning(f"[MOO3] Invalid TP {ind.sell_pct:+.4f} -> set to 0.06")
            ind.sell_pct = 0.06
        if ind.sell_days < 3 or ind.sell_days > 60:
            logger.warning(f"[MOO3] Invalid days {ind.sell_days} -> set to 15")
            ind.sell_days = 15
        return ind


# ── Convenience loader (for main.py integration) ─────────────────────────────

def load_and_register_moo3(df: pd.DataFrame, weight: float = 2.0) -> bool:
    # ── Guard: skip if already registered ────────────────────────────
    from strategies.strategy_engine import list_strategies
    if "MOO3" in list_strategies():
        return True   # already loaded on a previous ticker — skip silently

    if not os.path.exists(_MODEL_PATH):
        return False
    try:
        best = MOO3Engine.load()
        best = MOO3Engine.sanitize_moo3_params(best)
        from strategies.strategy_engine import register_strategy

        @register_strategy("MOO3", weight=weight)
        def moo3_plugin(df_: pd.DataFrame) -> str:
            return best.predict(df_)

        print(f"  [MOO3] Loaded from {_MODEL_PATH} — registered as plugin")
        return True
    except Exception as e:
        print(f"  [MOO3] Failed to load model: {e}")
        return False
