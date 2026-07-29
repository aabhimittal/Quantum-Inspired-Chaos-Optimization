"""Feasibility constraints on failure configurations.

Real industrial systems cannot fail in arbitrary ways.  Some fault combinations
are physically impossible ("a node cannot be both *partitioned* and *the elected
leader*"), operators care only about failures within a **blast radius** ("what is
the worst case if at most 3 things break at once?"), and each fault carries an
occurrence cost that must fit an operational budget.

A :class:`ConstraintSet` captures these as first-class objects so the search
only explores *reachable* catastrophes.  Constraints support three things every
optimizer needs:

* :meth:`is_feasible` / :meth:`violations` — test a configuration,
* :meth:`repair` — nudge an infeasible configuration into the feasible region,
* :meth:`feasible_random` — draw a uniform-ish feasible configuration directly,
  which keeps sampling efficient even when the feasible region is tiny.

Constraints are expressed over **factor indices** (0..N-1).  Use
:meth:`ConstraintSet.build` to specify them with factor *names* against a
:class:`~quantum_chaos.core.search_space.SearchSpace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Config = Tuple[int, ...]


@dataclass
class ConstraintSet:
    """A collection of feasibility rules over ``num_factors`` binary factors."""

    num_factors: int
    max_active: Optional[int] = None
    min_active: Optional[int] = None
    forbidden_pairs: List[Tuple[int, int]] = field(default_factory=list)
    mutual_exclusions: List[List[int]] = field(default_factory=list)
    implications: List[Tuple[int, int]] = field(default_factory=list)
    required: List[int] = field(default_factory=list)
    weights: Optional[np.ndarray] = None
    budget: Optional[float] = None
    predicates: List[Callable[[Config], bool]] = field(default_factory=list)
    #: human-readable labels for the custom predicates (parallel to ``predicates``)
    predicate_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != (self.num_factors,):
                raise ValueError(
                    f"weights must have shape ({self.num_factors},), "
                    f"got {self.weights.shape}"
                )
        if self.budget is not None and self.weights is None:
            # Default to unit weights so a budget alone acts like max_active.
            self.weights = np.ones(self.num_factors, dtype=float)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        space,
        max_active: Optional[int] = None,
        min_active: Optional[int] = None,
        forbidden_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        mutual_exclusions: Optional[Sequence[Sequence[str]]] = None,
        implications: Optional[Sequence[Tuple[str, str]]] = None,
        required: Optional[Sequence[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        budget: Optional[float] = None,
    ) -> "ConstraintSet":
        """Build a constraint set using factor *names* from a ``SearchSpace``."""
        index = {name: i for i, name in enumerate(space.names)}

        def idx(name: str) -> int:
            if name not in index:
                raise KeyError(f"unknown factor {name!r}; have {list(index)}")
            return index[name]

        w = None
        if weights is not None:
            w = np.ones(space.num_factors, dtype=float)
            for name, val in weights.items():
                w[idx(name)] = float(val)

        return cls(
            num_factors=space.num_factors,
            max_active=max_active,
            min_active=min_active,
            forbidden_pairs=[(idx(a), idx(b)) for a, b in (forbidden_pairs or [])],
            mutual_exclusions=[[idx(n) for n in grp] for grp in (mutual_exclusions or [])],
            implications=[(idx(a), idx(b)) for a, b in (implications or [])],
            required=[idx(n) for n in (required or [])],
            weights=w,
            budget=budget,
        )

    # -- feasibility ------------------------------------------------------
    def cost(self, bits: Config) -> float:
        """Total weighted occurrence cost of the active factors."""
        if self.weights is None:
            return float(sum(bits))
        return float(np.asarray(bits, dtype=float) @ self.weights)

    def violations(self, bits: Config) -> List[str]:
        """Human-readable list of every rule ``bits`` breaks (empty = feasible)."""
        out: List[str] = []
        active = int(sum(bits))
        if self.max_active is not None and active > self.max_active:
            out.append(f"max_active: {active} > {self.max_active}")
        if self.min_active is not None and active < self.min_active:
            out.append(f"min_active: {active} < {self.min_active}")
        for i, j in self.forbidden_pairs:
            if bits[i] and bits[j]:
                out.append(f"forbidden_pair: ({i},{j}) both active")
        for grp in self.mutual_exclusions:
            if sum(bits[k] for k in grp) > 1:
                out.append(f"mutual_exclusion: >1 active in {grp}")
        for a, b in self.implications:
            if bits[a] and not bits[b]:
                out.append(f"implication: {a} active requires {b}")
        for r in self.required:
            if not bits[r]:
                out.append(f"required: {r} must be active")
        if self.budget is not None and self.cost(bits) > self.budget + 1e-9:
            out.append(f"budget: cost {self.cost(bits):.3g} > {self.budget:.3g}")
        for name, pred in zip(self._predicate_labels(), self.predicates):
            try:
                ok = bool(pred(tuple(bits)))
            except Exception as exc:  # a broken predicate is a violation, not a crash
                out.append(f"predicate {name!r} raised: {exc}")
                continue
            if not ok:
                out.append(f"predicate: {name}")
        return out

    def is_feasible(self, bits: Config) -> bool:
        return not self.violations(bits)

    def _predicate_labels(self) -> List[str]:
        if self.predicate_names and len(self.predicate_names) == len(self.predicates):
            return list(self.predicate_names)
        return [f"custom_{i}" for i in range(len(self.predicates))]

    # -- repair & sampling ------------------------------------------------
    def repair(self, bits: Config, rng: Optional[np.random.Generator] = None) -> Config:
        """Greedily transform ``bits`` toward feasibility.

        Order: satisfy required/implications (turn things on), resolve
        exclusions and forbidden pairs (turn the cheaper factor off), then trim
        by cost/cardinality (drop highest-cost actives).  Best-effort — returns
        the closest feasible configuration it can reach.
        """
        rng = rng or np.random.default_rng()
        b = list(int(x) for x in bits)
        w = self.weights if self.weights is not None else np.ones(self.num_factors)

        # 1) Required factors on.
        for r in self.required:
            b[r] = 1
        # 2) Implications: a -> b.
        for _ in range(self.num_factors):  # iterate to a fixed point
            changed = False
            for a, imp in self.implications:
                if b[a] and not b[imp]:
                    b[imp] = 1
                    changed = True
            if not changed:
                break
        # 3) Mutual exclusions: keep the lowest-cost member, drop the rest.
        for grp in self.mutual_exclusions:
            actives = [k for k in grp if b[k]]
            if len(actives) > 1:
                keep = min(actives, key=lambda k: w[k])
                for k in actives:
                    if k != keep:
                        b[k] = 0
        # 4) Forbidden pairs: drop the higher-cost (non-required) member.
        for i, j in self.forbidden_pairs:
            if b[i] and b[j]:
                drop = i if (w[i] >= w[j] and i not in self.required) else j
                if drop in self.required:
                    drop = i if drop == j else j
                b[drop] = 0
        # 5) Trim by budget / max_active: remove highest-cost, non-required actives.
        def actives_sorted():
            act = [k for k in range(self.num_factors) if b[k] and k not in self.required]
            return sorted(act, key=lambda k: -w[k])

        if self.max_active is not None:
            while sum(b) > self.max_active:
                act = actives_sorted()
                if not act:
                    break
                b[act[0]] = 0
        if self.budget is not None:
            while self.cost(tuple(b)) > self.budget + 1e-9:
                act = actives_sorted()
                if not act:
                    break
                b[act[0]] = 0
        # 6) min_active: turn on cheapest inactive, feasible factors.
        if self.min_active is not None:
            while sum(b) < self.min_active:
                inactive = [k for k in range(self.num_factors) if not b[k]]
                if not inactive:
                    break
                inactive.sort(key=lambda k: w[k])
                b[inactive[0]] = 1
        return tuple(b)

    def feasible_random(
        self,
        rng: np.random.Generator,
        max_tries: int = 64,
    ) -> Config:
        """Draw a feasible configuration (rejection sampling, then repair)."""
        # Bias the density toward the cardinality/budget limits so we do not
        # always reject on max_active for tightly-constrained spaces.
        p = 0.5
        if self.max_active is not None and self.num_factors > 0:
            p = min(0.5, self.max_active / self.num_factors)
        for _ in range(max_tries):
            bits = tuple(int(x) for x in (rng.random(self.num_factors) < p))
            if self.is_feasible(bits):
                return bits
        repaired = self.repair(tuple(int(x) for x in (rng.random(self.num_factors) < p)), rng)
        return repaired

    def feasible_fraction(self, sample: int = 2000, rng=None) -> float:
        """Estimate the fraction of the space that is feasible (Monte Carlo)."""
        rng = rng or np.random.default_rng(0)
        n = self.num_factors
        if n <= 16:  # exact by enumeration
            from itertools import product

            total = 0
            feas = 0
            for bits in product((0, 1), repeat=n):
                total += 1
                if self.is_feasible(bits):
                    feas += 1
            return feas / total
        hits = 0
        for _ in range(sample):
            bits = tuple(int(x) for x in (rng.random(n) < 0.5))
            if self.is_feasible(bits):
                hits += 1
        return hits / sample
