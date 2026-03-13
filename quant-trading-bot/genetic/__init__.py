"""
genetic/ — MOO3 Genetic Programming Engine
============================================
Full implementation of the Long, Kampouridis & Papastylianou (2026)
MOO3 framework: NSGA-II multi-objective GP with Directional Changes.

Reference
---------
Long, X., Kampouridis, M., & Papastylianou, T. (2026).
"Multi-objective genetic programming-based algorithmic trading, using
directional changes and a modified Sharpe ratio score for identifying
optimal trading strategies." Artificial Intelligence Review, 59:39.
DOI: 10.1007/s10462-025-11390-9

Objectives (user-configured)
------------------------------
  f1 — Total Return      (MAXIMISE)
  f2 — Win Rate          (MAXIMISE)
  f3 — Max Drawdown      (MINIMISE)

Pipeline
--------
  run_genetic.py
       │
       ├─ gp_engine.py        ← MOO3Engine (main loop)
       │       ├─ gp_tree.py  ← Node, tree ops (crossover / mutation)
       │       ├─ terminals.py ← DC + TA indicator terminal set
       │       ├─ fitness.py  ← trade simulator → (TR, WR, MaxDD)
       │       ├─ nsga2.py    ← NSGA-II (sort, crowding, tournament)
       │       └─ sharpe_selector.py ← mSR Pareto picker
       │
       └─ Saves best individual → models/moo3_best.pkl
          Registers as "MOO3" plugin in strategy_engine
"""

from genetic.gp_engine import MOO3Engine, MOO3Individual  # noqa