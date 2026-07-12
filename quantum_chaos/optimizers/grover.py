"""Grover / amplitude-amplification search for catastrophic configurations.

Grover's algorithm shines when you can *recognize* a catastrophic failure
(``failure >= threshold``) but the configuration space is far too large to scan.
Given such a recognizer (oracle) marking ``M`` catastrophic states out of
``N = 2**n``, amplitude amplification surfaces one of them in ``~ (π/4)·√(N/M)``
oracle queries versus the ``~ N/M`` a classical random search needs — a
quadratic speedup.

In simulation the oracle is built from the target itself (enumerating the space
once), so this optimizer is meant for small/medium ``n``.  The headline metric
is *oracle queries*: the result's metadata reports the Grover query count beside
the classical expectation so the speedup is explicit and honest.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.problem import FailureProblem
from ..core.search_space import index_to_config
from ..quantum.backends import Backend, get_backend

try:  # pragma: no cover - import guard
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import grover_operator

    _HAS_QISKIT = True
except Exception:  # pragma: no cover
    _HAS_QISKIT = False


class GroverSearch:
    """Amplitude amplification toward the darkest (highest-failure) corners."""

    name = "grover"
    MAX_QUBITS = 16

    def __init__(
        self,
        quantile: Optional[float] = None,
        threshold: Optional[float] = None,
        shots: int = 512,
        backend: Optional[Backend] = None,
        backend_name: str = "statevector",
        max_evaluations: int = 1_000_000,
        seed: Optional[int] = None,
    ):
        if not _HAS_QISKIT:
            raise ImportError(
                "Qiskit is required for GroverSearch. Install with:\n"
                "    pip install 'quantum-chaos[quantum]'"
            )
        self.quantile = float(quantile) if quantile is not None else None
        self.threshold = float(threshold) if threshold is not None else None
        self.shots = int(shots)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.max_evaluations = int(max_evaluations)
        self.backend = backend or get_backend(backend_name, seed=seed)
        self.backend_name = getattr(self.backend, "name", backend_name)
        self._metadata: dict = {}

    def _build_oracle(self, marked_mask: np.ndarray, n: int) -> "QuantumCircuit":
        """Phase-flip oracle: multiplies marked basis states by -1."""
        from qiskit.circuit.library import DiagonalGate

        diag = np.where(marked_mask, -1.0, 1.0)
        oracle = QuantumCircuit(n)
        oracle.append(DiagonalGate(diag.tolist()), range(n))
        return oracle

    def _threshold(self, scores: np.ndarray) -> float:
        """Failure cutoff separating catastrophic configs from the rest.

        Explicit ``threshold`` wins; else a ``quantile`` if given; else a robust
        default halfway between the median and the maximum, which isolates a
        sparse "needle" that percentile methods would drown in a flat baseline.
        """
        if self.threshold is not None:
            return float(self.threshold)
        if self.quantile is not None:
            return float(np.quantile(scores, self.quantile))
        med = float(np.median(scores))
        mx = float(scores.max())
        return med + 0.5 * (mx - med) if mx > med else mx

    def optimize(self, problem: FailureProblem, reset: bool = True) -> "OptimizationResult":
        from ..core.result import OptimizationResult

        if reset:
            problem.reset_stats()
        n = problem.num_qubits
        if n > self.MAX_QUBITS:
            raise ValueError(
                f"GroverSearch enumerates the space to build its oracle; "
                f"n<={self.MAX_QUBITS} required, got {n}."
            )
        N = 2 ** n

        # Enumerate once to build the recognizer (oracle).
        scores = np.empty(N)
        for idx in range(N):
            scores[idx] = problem.evaluate(index_to_config(idx, n))

        threshold = self._threshold(scores)
        marked_mask = scores >= threshold
        M = int(marked_mask.sum())
        if M == 0:  # threshold too high; mark the single worst
            marked_mask[int(np.argmax(scores))] = True
            M = 1

        # Optimal number of Grover iterations.
        ratio = N / M
        iterations = max(1, int(np.floor((np.pi / 4.0) * np.sqrt(ratio))))
        # Guard against over-rotation when M is a large fraction of N.
        max_useful = max(1, int(np.floor((np.pi / 4.0) * np.sqrt(N))))
        iterations = min(iterations, max_useful)

        oracle = self._build_oracle(marked_mask, n)
        grover_op = grover_operator(oracle)

        circuit = QuantumCircuit(n)
        circuit.h(range(n))
        for _ in range(iterations):
            circuit.compose(grover_op, inplace=True)

        counts = self.backend.sample(circuit, self.shots)
        best_bits = max(counts, key=lambda c: scores[_config_index(c)])

        # Honest speedup accounting on oracle queries.
        classical_expected_queries = ratio  # ~N/M random draws to hit a marked state
        self._metadata = {
            "num_marked": M,
            "space_size": N,
            "threshold": threshold,
            "quantile": self.quantile,
            "grover_iterations": iterations,
            "grover_oracle_queries": iterations,
            "classical_expected_queries": classical_expected_queries,
            "speedup_factor": classical_expected_queries / max(1, iterations),
        }
        return problem.build_result(
            optimizer=self.name,
            backend=self.backend_name,
            metadata=self._metadata,
            best_bits=best_bits,
        )


def _config_index(bits) -> int:
    idx = 0
    for i, b in enumerate(bits):
        if b:
            idx |= 1 << i
    return idx
