"""Quantum-inspired simulated annealing (transverse-field flavored).

A classical annealer that borrows one idea from quantum annealing: a
*transverse-field* schedule ``Γ(t)`` that starts high and decays to zero.  While
``Γ`` is high the annealer performs multi-bit "tunneling" moves that can jump
across tall, thin barriers in the failure landscape (escaping deceptive traps);
as ``Γ`` decays it settles into local single-bit refinement.  Combined with a
cooling temperature ``T(t)`` and Metropolis acceptance, this is a strong,
dependency-free baseline that contrasts with the true-quantum optimizers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.problem import FailureProblem
from .base import Optimizer


class SimulatedQuantumAnnealer(Optimizer):
    """Transverse-field-inspired simulated annealing over failure configs."""

    name = "annealing"

    def __init__(
        self,
        t_start: float = 1.0,
        t_end: float = 0.02,
        gamma_start: float = 0.6,
        gamma_end: float = 0.0,
        max_evaluations: int = 2000,
        seed: Optional[int] = None,
    ):
        super().__init__(max_evaluations=max_evaluations, seed=seed)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.gamma_start = float(gamma_start)
        self.gamma_end = float(gamma_end)
        self._metadata: dict = {}

    def _tunnel(self, bits: np.ndarray, gamma: float) -> np.ndarray:
        """Propose a neighbour; with prob ~gamma flip several bits (tunnel)."""
        n = bits.shape[0]
        proposal = bits.copy()
        if self.rng.random() < gamma:
            # Multi-bit tunneling move: flip k bits, k drawn from gamma.
            k = 1 + self.rng.integers(1, max(2, int(1 + gamma * n)))
            k = min(k, n)
            idxs = self.rng.choice(n, size=k, replace=False)
            proposal[idxs] ^= 1
        else:
            idx = self.rng.integers(0, n)
            proposal[idx] ^= 1
        return proposal

    def _run(self, problem: FailureProblem) -> Optional[tuple]:
        n = problem.num_qubits
        steps = self.max_evaluations
        bits = np.array(problem.search_space.random_config(self.rng), dtype=int)
        score = problem.evaluate(tuple(bits))
        best_bits, best_score = bits.copy(), score

        step = 0
        while self.budget_left(problem):
            frac = step / max(1, steps - 1)
            T = self.t_start * (self.t_end / self.t_start) ** frac
            gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * frac

            proposal = self._tunnel(bits, gamma)
            new_score = problem.evaluate(tuple(proposal))
            delta = new_score - score
            if delta >= 0 or self.rng.random() < np.exp(delta / max(T, 1e-9)):
                bits, score = proposal, new_score
                if score > best_score:
                    best_bits, best_score = bits.copy(), score
            step += 1

        self._metadata = {
            "steps": step,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "gamma_start": self.gamma_start,
        }
        return tuple(int(b) for b in best_bits)
