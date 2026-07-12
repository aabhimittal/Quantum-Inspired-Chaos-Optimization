"""Optimizer base class and a shared SPSA routine.

Every optimizer consumes a :class:`~quantum_chaos.core.problem.FailureProblem`
and returns an :class:`~quantum_chaos.core.result.OptimizationResult`.  The base
class handles the evaluation budget, book-keeping, and result assembly so the
concrete optimizers can focus on their search strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from ..core.problem import FailureProblem
from ..core.result import OptimizationResult


class Optimizer(ABC):
    """Abstract failure-mode optimizer."""

    #: label used in reports / results
    name: str = "optimizer"

    def __init__(self, max_evaluations: int = 2000, seed: Optional[int] = None):
        self.max_evaluations = int(max_evaluations)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def budget_left(self, problem: FailureProblem) -> bool:
        # Budget is measured in *queries* (every evaluate call), so a search
        # that exhausts a small space's distinct configs still terminates.
        return problem.num_queries < self.max_evaluations

    @abstractmethod
    def _run(self, problem: FailureProblem) -> Optional[tuple]:
        """Search ``problem``; optionally return the best bit tuple found."""

    def optimize(self, problem: FailureProblem, reset: bool = True) -> OptimizationResult:
        if reset:
            problem.reset_stats()
        best_bits = self._run(problem)
        return problem.build_result(
            optimizer=self.name,
            backend=getattr(self, "backend_name", ""),
            metadata=getattr(self, "_metadata", {}),
            best_bits=best_bits,
        )


def spsa_maximize(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    a: float = 0.2,
    c: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
    should_stop: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Maximize a noisy ``objective`` with SPSA; return the best parameters.

    Simultaneous Perturbation Stochastic Approximation estimates the gradient
    from just two objective evaluations per step regardless of dimension, which
    is exactly what we want when each evaluation means running a quantum circuit
    and scoring samples on the target.
    """
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_val = objective(x)
    A = max(1.0, 0.1 * iterations)
    for k in range(iterations):
        if should_stop is not None and should_stop():
            break
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.shape)
        f_plus = objective(x + ck * delta)
        f_minus = objective(x - ck * delta)
        ghat = (f_plus - f_minus) / (2.0 * ck) * delta  # ascent direction
        x = x + ak * ghat
        val = objective(x)
        if val > best_val:
            best_val, best_x = val, x.copy()
    return best_x
