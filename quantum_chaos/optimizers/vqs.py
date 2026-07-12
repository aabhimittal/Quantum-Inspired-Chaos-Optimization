"""Variational Quantum Sampler (VQS) — the flagship failure-mode optimizer.

The intuition from the project's premise made literal:

1. A parameterized circuit prepares a *probability wave* (superposition) over
   every possible failure configuration.
2. We measure it, score the sampled configurations on the target system, and
   ask "how bad is the worst tail of what we drew?" (a CVaR objective).
3. A classical optimizer (SPSA) nudges the circuit parameters so amplitude
   **concentrates on the darkest corners** — the configurations most likely to
   break the system.
4. Measuring the trained circuit *collapses* the wave onto those worst-case
   failure combinations.

Unlike QAOA/Grover this works on *any* black-box target — no QUBO structure or
recognizer oracle required — and it depends only on core Qiskit primitives, so
it is robust across Qiskit versions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.problem import FailureProblem
from ..quantum.backends import Backend, get_backend, sample_configs
from ..quantum.circuits import bind, variational_ansatz
from .base import Optimizer, spsa_maximize


class VariationalQuantumSampler(Optimizer):
    """CVaR-based variational search for worst-case failure combinations."""

    name = "vqs"

    def __init__(
        self,
        reps: int = 2,
        shots: int = 256,
        iterations: int = 60,
        cvar_alpha: float = 0.15,
        entanglement: str = "linear",
        backend: Optional[Backend] = None,
        backend_name: str = "statevector",
        max_evaluations: int = 4000,
        seed: Optional[int] = None,
    ):
        super().__init__(max_evaluations=max_evaluations, seed=seed)
        self.reps = int(reps)
        self.shots = int(shots)
        self.iterations = int(iterations)
        self.cvar_alpha = float(cvar_alpha)
        self.entanglement = entanglement
        self.backend = backend or get_backend(backend_name, seed=seed)
        self.backend_name = getattr(self.backend, "name", backend_name)
        self._metadata: dict = {}

    def _cvar(self, samples: List[Tuple[tuple, float]], problem: FailureProblem) -> float:
        """Upper-tail CVaR of failure over the sampled configs."""
        scored = [(problem.evaluate(cfg), prob) for cfg, prob in samples]
        scored.sort(key=lambda t: t[0], reverse=True)  # worst (highest) first
        alpha = self.cvar_alpha
        acc_p = 0.0
        acc_val = 0.0
        for val, p in scored:
            take = min(p, alpha - acc_p)
            if take <= 0:
                break
            acc_val += take * val
            acc_p += take
        if acc_p <= 0:
            return scored[0][0] if scored else 0.0
        return acc_val / acc_p

    def _run(self, problem: FailureProblem) -> Optional[tuple]:
        n = problem.num_qubits
        circuit, params = variational_ansatz(n, reps=self.reps, entanglement=self.entanglement)

        def objective(theta: np.ndarray) -> float:
            bound = bind(circuit, params, theta)
            samples = sample_configs(self.backend, bound, self.shots)
            return self._cvar(samples, problem)

        x0 = self.rng.uniform(0.0, 2.0 * np.pi, size=len(params))
        initial_cvar = objective(x0)

        best_theta = spsa_maximize(
            objective,
            x0,
            iterations=self.iterations,
            rng=self.rng,
            should_stop=lambda: not self.budget_left(problem),
        )
        final_cvar = objective(best_theta)

        self._metadata = {
            "reps": self.reps,
            "shots": self.shots,
            "iterations": self.iterations,
            "cvar_alpha": self.cvar_alpha,
            "initial_cvar": initial_cvar,
            "final_cvar": final_cvar,
        }
        # Let the base class pick the single worst configuration ever measured.
        return None
