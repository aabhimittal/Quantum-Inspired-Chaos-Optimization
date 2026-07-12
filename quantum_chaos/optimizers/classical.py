"""Classical baselines for honest benchmarking.

If a quantum-inspired method cannot beat plain random search or a hill-climber
on failure discovery, that is worth knowing.  These baselines make the
comparison explicit; the ``benchmark`` CLI command reports them side by side.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.problem import FailureProblem
from .base import Optimizer


class RandomSearch(Optimizer):
    """Uniformly sample configurations until the budget is spent."""

    name = "random"

    def _run(self, problem: FailureProblem) -> Optional[tuple]:
        space = problem.search_space
        while self.budget_left(problem):
            problem.evaluate(space.random_config(self.rng))
        return None


class HillClimb(Optimizer):
    """(1+1)-EA: mutate each bit with prob ``1/n``, keep improvements.

    A strong local-search baseline.  It excels on smooth landscapes and gets
    stuck on deceptive ones — exactly the contrast the quantum methods target.
    """

    name = "hillclimb"

    def __init__(
        self,
        restarts: bool = True,
        max_evaluations: int = 2000,
        seed: Optional[int] = None,
    ):
        super().__init__(max_evaluations=max_evaluations, seed=seed)
        self.restarts = restarts

    def _run(self, problem: FailureProblem) -> Optional[tuple]:
        n = problem.num_qubits
        space = problem.search_space
        rng = self.rng

        bits = np.array(space.random_config(rng), dtype=int)
        score = problem.evaluate(tuple(bits))
        best_bits, best_score = bits.copy(), score
        stagnation = 0

        while self.budget_left(problem):
            flips = rng.random(n) < (1.0 / n)
            if not flips.any():
                flips[rng.integers(0, n)] = True
            candidate = bits.copy()
            candidate[flips] ^= 1
            cand_score = problem.evaluate(tuple(candidate))
            if cand_score >= score:
                if cand_score > score:
                    stagnation = 0
                bits, score = candidate, cand_score
                if score > best_score:
                    best_bits, best_score = bits.copy(), score
            else:
                stagnation += 1

            # Random restart when clearly stuck (helps on deceptive traps).
            if self.restarts and stagnation > 5 * n:
                bits = np.array(space.random_config(rng), dtype=int)
                score = problem.evaluate(tuple(bits))
                stagnation = 0

        return tuple(int(b) for b in best_bits)
