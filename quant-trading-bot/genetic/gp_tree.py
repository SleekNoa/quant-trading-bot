"""
genetic/gp_tree.py — Strongly-Typed GP Tree
=============================================
Implements the tree representation from Long et al. (2026) Section 4.

Structure
---------
    Root (fixed): ITE(condition → BUY | HOLD)
    Evolved part: boolean expression using AND / OR / GT / LT
    Leaves:       TERMINAL (indicator name) | ERC (random constant [0,1])

Tree grammar
------------
    bool_expr := AND(bool_expr, bool_expr)
               | OR (bool_expr, bool_expr)
               | GT (TERMINAL, ERC)      # indicator > threshold
               | LT (TERMINAL, ERC)      # indicator < threshold

Genetic operators
-----------------
    subtree_crossover  — exchanges boolean subtrees between two parents
    point_mutation     — randomly changes a single node (paper Section 4)

Initialisation
--------------
    ramped_half_and_half — mix of "full" and "grow" trees across depth range
                           (standard GP initialisation, Koza 1992)
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple


# ── Node ──────────────────────────────────────────────────────────────────────

class Node:
    """
    A single node in a GP boolean expression tree.

    ntype : "AND" | "OR" | "GT" | "LT" | "TERMINAL" | "ERC"
    children : list of child Nodes
    value    : str (terminal name) or float (ERC value)
    """

    __slots__ = ("ntype", "children", "value")

    def __init__(
        self,
        ntype: str,
        children: Optional[List["Node"]] = None,
        value=None,
    ) -> None:
        self.ntype    = ntype
        self.children = children if children is not None else []
        self.value    = value

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, indicators: Dict[str, float]) -> bool:
        """
        Recursively evaluate this boolean expression.

        indicators : dict {terminal_name: normalised_float [0,1]}
        """
        n = self.ntype
        if n == "AND":
            return self.children[0].evaluate(indicators) and self.children[1].evaluate(indicators)
        if n == "OR":
            return self.children[0].evaluate(indicators) or  self.children[1].evaluate(indicators)
        if n in ("GT", "LT"):
            ind_val = indicators.get(self.children[0].value, 0.5)
            erc_val = self.children[1].value
            return ind_val > erc_val if n == "GT" else ind_val < erc_val
        # TERMINAL / ERC leaves should not be reached as root of evaluation
        return False

    # ── Tree traversal ────────────────────────────────────────────────────────

    def all_nodes(self) -> List["Node"]:
        """Pre-order traversal returning all nodes."""
        result: List[Node] = [self]
        for child in self.children:
            result.extend(child.all_nodes())
        return result

    def size(self) -> int:
        return len(self.all_nodes())

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def clone(self) -> "Node":
        return copy.deepcopy(self)

    # ── Human-readable repr ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.ntype in ("AND", "OR"):
            return f"({self.children[0]} {self.ntype} {self.children[1]})"

        if self.ntype in ("GT", "LT"):
            left = self.children[0].value
            right = self.children[1].value
            op = ">" if self.ntype == "GT" else "<"

            # format only numeric values
            def fmt(val):
                return f"{val:.3f}" if isinstance(val, (float, int)) else str(val)

            return f"({fmt(left)} {op} {fmt(right)})"

        if self.ntype == "ERC":
            return f"{self.value:.3f}" if isinstance(self.value, (float, int)) else str(self.value)

        return str(self.value)  # TERMINAL


# ── Leaf constructors ─────────────────────────────────────────────────────────

def _terminal(names: List[str]) -> Node:
    return Node("TERMINAL", value=random.choice(names))


def _erc() -> Node:
    return Node("ERC", value=random.random())


def _comparison(names: List[str]) -> Node:
    """GT or LT comparison: TERMINAL OP ERC."""
    op = random.choice(["GT", "LT"])
    return Node(op, children=[_terminal(names), _erc()])


# ── Tree generator ────────────────────────────────────────────────────────────

def random_tree(
    terminal_names: List[str],
    max_depth: int = 4,
    method: str = "grow",
) -> Node:
    """
    Recursively generate a random boolean expression tree.

    method = "full" → always build to max_depth (denser trees)
    method = "grow" → stop randomly before max_depth (varied shapes)
    """
    if max_depth <= 0:
        return _comparison(terminal_names)

    if method == "grow":
        # ~40% chance to terminate early with a comparison leaf
        if random.random() < 0.40:
            return _comparison(terminal_names)

    op   = random.choice(["AND", "OR"])
    left  = random_tree(terminal_names, max_depth - 1, method)
    right = random_tree(terminal_names, max_depth - 1, method)
    return Node(op, children=[left, right])


def ramped_half_and_half(
    terminal_names: List[str],
    pop_size: int,
    max_depth: int = 5,
) -> List[Node]:
    """
    Koza (1992) initialisation: mix of full + grow trees at varying depths.
    Ensures initial population diversity.
    """
    trees: List[Node] = []
    min_depth = 2
    depth_range = list(range(min_depth, max_depth + 1))
    per_cell = max(1, pop_size // (len(depth_range) * 2))

    for d in depth_range:
        for _ in range(per_cell):
            trees.append(random_tree(terminal_names, d, "full"))
        for _ in range(per_cell):
            trees.append(random_tree(terminal_names, d, "grow"))

    # Fill any remaining slots with grow trees
    while len(trees) < pop_size:
        trees.append(random_tree(terminal_names, max_depth, "grow"))

    return trees[:pop_size]


# ── Helper: find node by pre-order index ─────────────────────────────────────

def _find_node(root: Node, target_idx: int) -> Tuple[Optional[Node], Optional[int], Optional[Node]]:
    """
    Return (parent, child_slot_index, target_node) for node at target_idx.
    Returns (None, None, root) when target_idx == 0.
    """
    counter = [0]
    result: List = [None]

    def _traverse(node: Node, parent: Optional[Node] = None, slot: Optional[int] = None):
        if result[0] is not None:
            return
        idx = counter[0]
        counter[0] += 1
        if idx == target_idx:
            result[0] = (parent, slot, node)
            return
        for i, child in enumerate(node.children):
            _traverse(child, node, i)

    _traverse(root)
    return result[0] if result[0] is not None else (None, None, None)


# ── Genetic operators ─────────────────────────────────────────────────────────

def subtree_crossover(
    tree1: Node,
    tree2: Node,
    max_result_depth: int = 7,
) -> Tuple[Node, Node]:
    """
    Subtree crossover (Long et al. 2026, Section 4).

    Randomly select a crossover point in each tree (excluding root to
    preserve ITE structure) and swap the subtrees.  If swapping would
    exceed max_result_depth the operation is aborted and the originals
    are returned unchanged (bloat control).
    """
    t1 = tree1.clone()
    t2 = tree2.clone()

    n1, n2 = t1.size(), t2.size()
    if n1 < 2 or n2 < 2:
        return t1, t2

    # Pick internal (non-root) crossover points
    idx1 = random.randint(1, n1 - 1)
    idx2 = random.randint(1, n2 - 1)

    parent1, slot1, sub1 = _find_node(t1, idx1)
    parent2, slot2, sub2 = _find_node(t2, idx2)

    if parent1 is None or parent2 is None:
        return t1, t2
    if sub1 is None or sub2 is None:
        return t1, t2

    # Bloat guard
    new_depth1 = sub2.depth() + _parent_depth(t1, idx1)
    new_depth2 = sub1.depth() + _parent_depth(t2, idx2)
    if new_depth1 > max_result_depth or new_depth2 > max_result_depth:
        return t1, t2

    parent1.children[slot1] = sub2.clone()
    parent2.children[slot2] = sub1.clone()
    return t1, t2


def _parent_depth(root: Node, child_idx: int) -> int:
    """Estimate depth of the parent of the node at child_idx."""
    # Simplified: compute total depth up to that node's parent
    counter = [0]
    depth_result = [0]
    found = [False]

    def _traverse(node: Node, current_depth: int):
        if found[0]:
            return
        idx = counter[0]
        counter[0] += 1
        if idx == child_idx:
            depth_result[0] = current_depth
            found[0] = True
            return
        for child in node.children:
            _traverse(child, current_depth + 1)

    _traverse(root, 0)
    return depth_result[0]


def point_mutation(tree: Node, terminal_names: List[str]) -> Node:
    """
    Point mutation (Long et al. 2026, Section 4).

    Randomly select one non-root node and replace it with a new
    compatible random element:
      - AND/OR  → flip to the other boolean operator
      - GT/LT   → flip operator, replace terminal, or replace ERC
      - ERC     → sample new random constant
      - TERMINAL→ sample new random terminal
    """
    t = tree.clone()
    nodes = t.all_nodes()

    if len(nodes) < 2:
        return t

    # Select a random non-root node
    target = random.choice(nodes[1:])

    if target.ntype in ("AND", "OR"):
        target.ntype = "OR" if target.ntype == "AND" else "AND"

    elif target.ntype in ("GT", "LT"):
        mutation = random.choice(["flip_op", "new_terminal", "new_erc"])
        if mutation == "flip_op":
            target.ntype = "LT" if target.ntype == "GT" else "GT"
        elif mutation == "new_terminal":
            target.children[0].value = random.choice(terminal_names)
        else:
            target.children[1].value = random.random()

    elif target.ntype == "ERC":
        # Creep mutation: small perturbation 50% of time, full resample otherwise
        if random.random() < 0.5:
            target.value = float(np.clip(
                target.value + random.gauss(0, 0.1), 0.0, 1.0
            ))
        else:
            target.value = random.random()

    elif target.ntype == "TERMINAL":
        target.value = random.choice(terminal_names)

    return t


# ── Needed for ERC creep mutation ────────────────────────────────────────────
import numpy as np  # noqa: E402