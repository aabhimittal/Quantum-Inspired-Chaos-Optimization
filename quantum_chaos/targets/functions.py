"""A library of synthetic *failure landscapes*.

Each landscape is a :class:`~quantum_chaos.targets.base.TargetSystem` with a
known worst-case configuration, so tests and benchmarks can check that an
optimizer actually located the darkest corner.  They are deliberately shaped to
exercise the properties that make real AI-system failure discovery hard:

* **DeceptiveTrap** — a strong local attractor pulls naive search toward a safe
  region while the true catastrophe hides at the opposite corner.
* **NeedleInHaystack** — a single hidden combination is catastrophic and
  everything else is flat: pure search difficulty (ideal for amplitude
  amplification).
* **CorrelatedFaults** — individual faults are harmless but specific *pairs*
  interact to produce disaster.  This is the central thesis of the project:
  worst-case behaviour lives in *combinations*, not single knobs.
* **SumThreshold** — a smooth monotone landscape (sanity baseline); the system
  degrades gracefully then collapses past a threshold.
* **MaxSatFailure** — a planted weighted MAX-2-SAT instance: failure = weight of
  violated robustness constraints.
* **CascadingFailure** — an outage that *propagates* across a service dependency
  graph; failure = blast radius.  A realistic partner for cardinality/budget
  constraints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import TargetSystem


def _bits(config: Dict[str, int], names: Sequence[str]) -> np.ndarray:
    return np.array([config[n] for n in names], dtype=int)


class DeceptiveTrap(TargetSystem):
    """Concatenated deceptive trap: catastrophe hides opposite the easy optimum.

    The factors are split into blocks of size ``k``.  Within a block, failure is
    highest when *all* bits are on (the catastrophe), but there is a deceptive
    ridge that rewards turning bits *off* — so greedy/naive search is lured to
    the all-zeros safe corner.  The global worst config is all-ones.
    """

    def __init__(self, num_factors: int = 8, k: int = 4, name: str = "deceptive_trap"):
        if num_factors % k != 0:
            raise ValueError(f"num_factors ({num_factors}) must be a multiple of k ({k})")
        self.num_factors = int(num_factors)
        self.k = int(k)
        self.name = name

    def _trap(self, u: int) -> float:
        # u = number of ones in the block (unitation).
        if u == self.k:
            return float(self.k)  # global catastrophe for the block
        return float(self.k - 1 - u)  # deceptive slope toward all-zeros

    def failure_score(self, config: Dict[str, int]) -> float:
        bits = _bits(config, self.make_search_space().names)
        total = 0.0
        for start in range(0, self.num_factors, self.k):
            block = bits[start : start + self.k]
            total += self._trap(int(block.sum()))
        return total

    def worst_config(self) -> Optional[Dict[str, int]]:
        names = self.make_search_space().names
        return {n: 1 for n in names}


class NeedleInHaystack(TargetSystem):
    """A single hidden catastrophic combination; everything else is flat.

    Failure is ``base`` almost everywhere and ``base + spike`` at exactly one
    configuration (the needle), with a faint Hamming-distance gradient so that
    gradient-free samplers still get a whisper of signal.  This is the hardest
    landscape for random search and the natural showcase for Grover.
    """

    def __init__(
        self,
        num_factors: int = 10,
        needle: Optional[Sequence[int]] = None,
        base: float = 0.05,
        spike: float = 1.0,
        gradient: float = 0.0,
        seed: int = 7,
        name: str = "needle",
    ):
        self.num_factors = int(num_factors)
        self.base = float(base)
        self.spike = float(spike)
        self.gradient = float(gradient)
        self.name = name
        if needle is None:
            rng = np.random.default_rng(seed)
            needle = rng.integers(0, 2, size=self.num_factors)
        self._needle = np.array(list(needle), dtype=int)
        if self._needle.shape[0] != self.num_factors:
            raise ValueError("needle length must equal num_factors")

    def failure_score(self, config: Dict[str, int]) -> float:
        bits = _bits(config, self.make_search_space().names)
        hamming = int(np.sum(bits != self._needle))
        if hamming == 0:
            return self.base + self.spike
        return self.base + self.gradient * (self.num_factors - hamming) / self.num_factors

    def worst_config(self) -> Optional[Dict[str, int]]:
        names = self.make_search_space().names
        return {n: int(b) for n, b in zip(names, self._needle)}


class CorrelatedFaults(TargetSystem):
    """Single faults are cheap; specific *pairs* interact catastrophically.

    Failure = small linear cost per active fault + large positive reward for
    each *activated interaction pair*.  Because the interaction terms dominate,
    the worst case is turning on exactly the faults that participate in pairs —
    a combination a one-factor-at-a-time analysis would never surface.
    """

    def __init__(
        self,
        num_factors: int = 8,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        pair_weight: float = 1.0,
        linear_cost: float = 0.1,
        seed: int = 3,
        num_pairs: Optional[int] = None,
        name: str = "correlated_faults",
    ):
        self.num_factors = int(num_factors)
        self.pair_weight = float(pair_weight)
        self.linear_cost = float(linear_cost)
        self.name = name
        rng = np.random.default_rng(seed)
        if pairs is None:
            if num_pairs is None:
                num_pairs = max(1, num_factors // 2)
            pairs = set()
            attempts = 0
            while len(pairs) < num_pairs and attempts < 1000:
                i, j = sorted(rng.choice(num_factors, size=2, replace=False))
                pairs.add((int(i), int(j)))
                attempts += 1
            pairs = sorted(pairs)
        self._pairs: List[Tuple[int, int]] = [tuple(p) for p in pairs]

    def failure_score(self, config: Dict[str, int]) -> float:
        bits = _bits(config, self.make_search_space().names)
        score = -self.linear_cost * float(bits.sum())
        for i, j in self._pairs:
            if bits[i] and bits[j]:
                score += self.pair_weight
        return score

    def worst_config(self) -> Optional[Dict[str, int]]:
        # Worst = activate every factor that appears in at least one pair,
        # since each satisfied pair outweighs its two linear costs by design
        # (pair_weight > 2 * linear_cost is assumed for the default params).
        names = self.make_search_space().names
        involved = set()
        for i, j in self._pairs:
            involved.add(i)
            involved.add(j)
        if self.pair_weight <= 2 * self.linear_cost:
            return None  # ground truth not obvious; skip validation
        return {n: (1 if idx in involved else 0) for idx, n in enumerate(names)}


class SumThreshold(TargetSystem):
    """Smooth monotone degradation: a graceful sanity-check landscape.

    Failure rises with the number of active faults following a logistic curve
    centred on ``threshold``.  Monotone, so the worst case is all faults on;
    useful to confirm an optimizer converges on easy problems too.
    """

    def __init__(
        self,
        num_factors: int = 8,
        threshold: Optional[float] = None,
        sharpness: float = 1.0,
        name: str = "sum_threshold",
    ):
        self.num_factors = int(num_factors)
        self.threshold = float(threshold) if threshold is not None else num_factors / 2.0
        self.sharpness = float(sharpness)
        self.name = name

    def failure_score(self, config: Dict[str, int]) -> float:
        bits = _bits(config, self.make_search_space().names)
        x = self.sharpness * (float(bits.sum()) - self.threshold)
        return float(1.0 / (1.0 + np.exp(-x)))

    def worst_config(self) -> Optional[Dict[str, int]]:
        names = self.make_search_space().names
        return {n: 1 for n in names}


class MaxSatFailure(TargetSystem):
    """Planted weighted MAX-2-SAT: failure = weight of violated constraints.

    Each clause is a robustness constraint ``(literal_a OR literal_b)`` that the
    system is supposed to satisfy.  A configuration that violates high-weight
    clauses breaks the system.  We *plant* a worst-case assignment that violates
    every clause, so the ground-truth darkest corner is known even though the
    landscape is rugged and full of local optima.
    """

    def __init__(
        self,
        num_factors: int = 10,
        num_clauses: Optional[int] = None,
        seed: int = 11,
        name: str = "maxsat",
    ):
        self.num_factors = int(num_factors)
        self.name = name
        rng = np.random.default_rng(seed)
        if num_clauses is None:
            num_clauses = 3 * num_factors
        # Plant a target assignment; build clauses each violated by it.
        self._planted = rng.integers(0, 2, size=num_factors)
        clauses: List[Tuple[int, bool, int, bool, float]] = []
        for _ in range(num_clauses):
            i, j = rng.choice(num_factors, size=2, replace=False)
            # A clause (x_i == sign_i) OR (x_j == sign_j) is violated by the
            # planted assignment iff both literals are false there.  Choose the
            # signs to be the opposite of the planted bits so the planted
            # assignment violates the clause.
            sign_i = bool(1 - self._planted[i])
            sign_j = bool(1 - self._planted[j])
            weight = float(rng.uniform(0.5, 1.5))
            clauses.append((int(i), sign_i, int(j), sign_j, weight))
        self._clauses = clauses

    def failure_score(self, config: Dict[str, int]) -> float:
        bits = _bits(config, self.make_search_space().names)
        violated = 0.0
        for i, sign_i, j, sign_j, weight in self._clauses:
            lit_i = bool(bits[i]) == sign_i
            lit_j = bool(bits[j]) == sign_j
            if not (lit_i or lit_j):
                violated += weight
        return violated

    def worst_config(self) -> Optional[Dict[str, int]]:
        names = self.make_search_space().names
        return {n: int(b) for n, b in zip(names, self._planted)}


class CascadingFailure(TargetSystem):
    """Cascading outage over a service dependency graph.

    Models the failure mode operators fear most: a few injected faults that
    *propagate*.  Each factor injects a fault at one service node; a node then
    goes down if it is injected directly or if a fraction ``threshold`` of its
    upstream dependencies are already down.  Failure = the number of nodes down
    once the cascade settles (the blast radius).

    Because propagation is super-additive, the worst case concentrates on a small
    set of high-leverage nodes — which makes this a realistic partner for
    cardinality / blast-radius constraints (see
    :class:`~quantum_chaos.core.constraints.ConstraintSet`).  There is no simple
    closed-form worst config, so :meth:`worst_config` returns ``None``.
    """

    def __init__(
        self,
        num_factors: int = 12,
        edge_prob: float = 0.25,
        threshold: float = 0.5,
        seed: int = 17,
        name: str = "cascading_failure",
    ):
        self.num_factors = int(num_factors)
        self.threshold = float(threshold)
        self.name = name
        rng = np.random.default_rng(seed)
        n = self.num_factors
        # Random DAG: edges only point to higher indices (acyclic propagation).
        self._upstream: List[List[int]] = [[] for _ in range(n)]
        for j in range(n):
            for i in range(j):
                if rng.random() < edge_prob:
                    self._upstream[j].append(i)

    def _cascade_size(self, injected: np.ndarray) -> int:
        n = self.num_factors
        down = injected.astype(bool).copy()
        # Nodes are in topological order (edges go low->high), so one forward
        # pass settles the cascade.
        for j in range(n):
            if down[j]:
                continue
            ups = self._upstream[j]
            if not ups:
                continue
            frac = sum(1 for i in ups if down[i]) / len(ups)
            if frac >= self.threshold:
                down[j] = True
        return int(down.sum())

    def failure_score(self, config: Dict[str, int]) -> float:
        bits = _bits(config, self.make_search_space().names)
        return float(self._cascade_size(bits))

    def worst_config(self) -> Optional[Dict[str, int]]:
        return None  # no closed-form ground truth
