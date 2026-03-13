"""
genetic/nsga2.py — NSGA-II Multi-Objective Optimisation
=========================================================
Pure-Python / NumPy implementation of the Non-dominated Sorting
Genetic Algorithm II (Deb et al. 2002), as used in Long et al. (2026).

Key operations
--------------
  fast_non_dominated_sort   — partition population into Pareto fronts
  crowding_distance         — measure solution density within a front
  tournament_select         — select mating parent using rank + crowding
  build_next_generation     — élite (+offspring) selection strategy

Reference
---------
  Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
  A fast and elitist multiobjective genetic algorithm: NSGA-II.
  IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

  Long et al. (2026) Section 4 — "Selection and genetic operators"
  "NSGA-II first compares the Pareto front rank; the individual with the
   lowest rank is selected. For equal rank, the one with the higher
   crowding distance is selected."

Fitness convention (this module)
---------------------------------
  objectives array shape : (pop_size, 3)
  column 0 : Total Return   — MAXIMISE  → stored as-is
  column 1 : Win Rate       — MAXIMISE  → stored as-is
  column 2 : Max Drawdown   — MINIMISE  → internally negated so that
                               all objectives are treated as maximisation
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


# ── Pareto dominance helpers ──────────────────────────────────────────────────

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Return True if solution a Pareto-dominates solution b.

    a dominates b iff:
      - a is not worse than b on ALL objectives (all ≥)
      - a is strictly better than b on AT LEAST ONE objective (some >)

    Both vectors are assumed to be in MAXIMISE form.
    """
    return bool(np.all(a >= b) and np.any(a > b))


def _to_max_form(objectives: np.ndarray) -> np.ndarray:
    """
    Convert to all-maximise form.
    Column 2 (Max Drawdown) is a minimisation objective → negate.
    """
    obj = objectives.copy().astype(np.float64)
    obj[:, 2] = -obj[:, 2]   # drawdown: lower is better → negate
    return obj


# ── Fast non-dominated sorting ────────────────────────────────────────────────

def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Deb et al. (2002) Algorithm 1.

    Parameters
    ----------
    objectives : (pop_size, 3) — raw objectives (TR, WR, MaxDD)

    Returns
    -------
    fronts : list of lists of indices, ordered by Pareto rank
             fronts[0] = rank-1 (best / non-dominated) front
             fronts[1] = rank-2 front, etc.
    """
    obj    = _to_max_form(objectives)
    n      = len(obj)
    S      = [[] for _ in range(n)]       # solutions dominated by i
    n_dom  = np.zeros(n, dtype=int)        # domination count for each i

    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(obj[i], obj[j]):
                S[i].append(j)
            elif _dominates(obj[j], obj[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front: List[int] = []
        for i in fronts[current_front]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)

    return fronts


# ── Crowding distance ─────────────────────────────────────────────────────────

def crowding_distance(
    front_indices: List[int],
    objectives:    np.ndarray,
) -> np.ndarray:
    """
    Compute crowding distance for all solutions in a single front.

    Long et al. (2026) Eq. (1):
      cd_i = Σ_x |f_x(i+1) - f_x(i-1)| / (f_x_max - f_x_min)

    Boundary solutions get distance = ∞ (always preferred).

    Returns
    -------
    distances : (pop_size,) array — distance for each solution in the
                FULL population (non-front solutions get 0)
    """
    n_obj = objectives.shape[1]
    n     = len(objectives)

    distances = np.zeros(n)

    if len(front_indices) <= 2:
        for idx in front_indices:
            distances[idx] = np.inf
        return distances

    front_obj = objectives[front_indices]   # (front_size, n_obj)

    for obj_idx in range(n_obj):
        sorted_order = np.argsort(front_obj[:, obj_idx])
        sorted_front = [front_indices[i] for i in sorted_order]
        sorted_vals  = front_obj[sorted_order, obj_idx]

        obj_min = sorted_vals[0]
        obj_max = sorted_vals[-1]
        denom   = obj_max - obj_min if abs(obj_max - obj_min) > 1e-12 else 1.0

        # Boundary solutions → infinite crowding distance
        distances[sorted_front[0]]  = np.inf
        distances[sorted_front[-1]] = np.inf

        for k in range(1, len(sorted_front) - 1):
            distances[sorted_front[k]] += (
                (sorted_vals[k + 1] - sorted_vals[k - 1]) / denom
            )

    return distances


def assign_ranks_and_distances(
    objectives: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: run full NSGA-II ranking.

    Returns
    -------
    ranks     : (pop_size,) int array — Pareto front rank (1 = best)
    distances : (pop_size,) float array — crowding distance
    """
    pop_size  = len(objectives)
    ranks     = np.zeros(pop_size, dtype=int)
    distances = np.zeros(pop_size, dtype=float)

    fronts = fast_non_dominated_sort(objectives)

    for rank, front in enumerate(fronts, start=1):
        for idx in front:
            ranks[idx] = rank
        front_dist = crowding_distance(front, objectives)
        distances  += front_dist

    return ranks, distances


# ── Tournament selection ──────────────────────────────────────────────────────

def tournament_select(
    ranks:     np.ndarray,
    distances: np.ndarray,
    k:         int = 3,
) -> int:
    """
    k-way tournament selection based on Pareto rank then crowding distance.

    Long et al. (2026):
    "After obtaining k random individuals from the population, NSGA-II
     first compares the Pareto front rank; the individual with the lowest
     rank wins. For equal rank, the one with the highest crowding distance
     is selected."

    Returns
    -------
    Index of selected individual.
    """
    pop_size = len(ranks)
    contestants = random.choices(range(pop_size), k=k)

    best = contestants[0]
    for c in contestants[1:]:
        if ranks[c] < ranks[best]:
            best = c
        elif ranks[c] == ranks[best] and distances[c] > distances[best]:
            best = c
    return best


# ── Next-generation selection (élite + strategy) ─────────────────────────────

def select_next_generation(
    objectives_combined: np.ndarray,
    pop_size:            int,
) -> np.ndarray:
    """
    NSGA-II (+ élite) selection strategy.

    Combines parents + offspring (2*pop_size individuals), sorts by
    Pareto rank and crowding distance, keeps top pop_size.

    Long et al. (2026):
    "The next step is to select the top P individuals from the combination
     of the new population and the old population, resulting in 2P individuals."

    Parameters
    ----------
    objectives_combined : (2*pop_size, 3) array for parents + offspring
    pop_size            : target population size

    Returns
    -------
    selected_indices : (pop_size,) int array — indices into combined array
    """
    total   = len(objectives_combined)
    ranks, distances = assign_ranks_and_distances(objectives_combined)

    selected: List[int] = []
    fronts = fast_non_dominated_sort(objectives_combined)

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            # Fill remaining slots using crowding distance (best spread)
            remaining = pop_size - len(selected)
            if remaining <= 0:
                break
            front_dist = crowding_distance(front, objectives_combined)
            front_sorted = sorted(front, key=lambda i: -front_dist[i])
            selected.extend(front_sorted[:remaining])
            break

    return np.array(selected, dtype=int)