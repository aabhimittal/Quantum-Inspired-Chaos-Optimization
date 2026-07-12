"""QAOA-based failure search over a QUBO surrogate of the target.

QAOA needs the objective as an Ising Hamiltonian, but real failure functions are
black boxes.  So we:

1. fit a quadratic **surrogate** of the failure landscape from random samples
   (:func:`~quantum_chaos.quantum.qubo.fit_qubo_surrogate`),
2. train the QAOA angles to concentrate probability on the surrogate's
   worst-case corner (done analytically on the exact circuit distribution — no
   extra target evaluations), then
3. **measure** the trained circuit and score those configurations on the *true*
   target, returning the genuinely worst one found.

The surrogate only steers the quantum exploration; the reported failure is
always measured on the real system.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.problem import FailureProblem
from ..core.search_space import index_to_config
from ..quantum.backends import Backend, get_backend
from ..quantum.circuits import bind_many, qaoa_ansatz
from ..quantum.qubo import QUBO, fit_qubo_surrogate
from .base import Optimizer, spsa_maximize


class QAOAOptimizer(Optimizer):
    """Quantum Approximate Optimization Algorithm on a fitted QUBO surrogate."""

    name = "qaoa"

    #: exact expected-energy training is only used at/below this qubit count
    MAX_EXACT_QUBITS = 18

    def __init__(
        self,
        reps: int = 2,
        shots: int = 512,
        iterations: int = 80,
        surrogate_samples: int = 150,
        final_shots: Optional[int] = None,
        backend: Optional[Backend] = None,
        backend_name: str = "statevector",
        max_evaluations: int = 4000,
        seed: Optional[int] = None,
    ):
        super().__init__(max_evaluations=max_evaluations, seed=seed)
        self.reps = int(reps)
        self.shots = int(shots)
        self.iterations = int(iterations)
        self.surrogate_samples = int(surrogate_samples)
        self.final_shots = int(final_shots) if final_shots else int(shots) * 4
        self.backend = backend or get_backend(backend_name, seed=seed)
        self.backend_name = getattr(self.backend, "name", backend_name)
        self._metadata: dict = {}

    def _surrogate_energies(self, qubo: QUBO, n: int) -> np.ndarray:
        """Vector of surrogate failure values for every basis state (small n)."""
        energies = np.empty(2 ** n)
        for idx in range(2 ** n):
            bits = index_to_config(idx, n)
            energies[idx] = qubo.energy(bits)
        return energies

    def _run(self, problem: FailureProblem) -> Optional[tuple]:
        n = problem.num_qubits
        if n > self.MAX_EXACT_QUBITS:
            raise ValueError(
                f"QAOAOptimizer uses exact surrogate training for n<={self.MAX_EXACT_QUBITS} "
                f"qubits; got {n}. Use the VQS optimizer for larger spaces."
            )

        # 1) Fit the quadratic surrogate (consumes surrogate_samples target evals).
        qubo = fit_qubo_surrogate(
            problem, num_samples=self.surrogate_samples, rng=self.rng
        )
        h, J, _offset = qubo.to_ising()
        circuit, gammas, betas = qaoa_ansatz(h, J, n, reps=self.reps)

        # 2) Train angles to maximize expected *surrogate* failure — analytic,
        #    no target evaluations spent here.
        energies = self._surrogate_energies(qubo, n)

        def objective(params: np.ndarray) -> float:
            g = params[: self.reps]
            b = params[self.reps :]
            bound = bind_many(circuit, [(gammas, g), (betas, b)])
            probs = self.backend.probabilities(bound)
            return float(probs @ energies)  # expected surrogate failure

        x0 = self.rng.uniform(0.0, np.pi, size=2 * self.reps)
        best_params = spsa_maximize(
            objective, x0, iterations=self.iterations, rng=self.rng
        )

        # 3) Measure the trained circuit and score on the TRUE target.
        g = best_params[: self.reps]
        b = best_params[self.reps :]
        trained = bind_many(circuit, [(gammas, g), (betas, b)])
        counts = self.backend.sample(trained, self.final_shots)
        for cfg in counts:
            if not self.budget_left(problem):
                break
            problem.evaluate(cfg)

        self._metadata = {
            "reps": self.reps,
            "surrogate_samples": self.surrogate_samples,
            "final_shots": self.final_shots,
            "distinct_final_configs": len(counts),
        }
        return None
