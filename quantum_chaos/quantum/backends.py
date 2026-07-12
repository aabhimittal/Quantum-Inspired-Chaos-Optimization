"""Execution backends — the bridge to real Qiskit.

Optimizers never talk to Qiskit primitives directly; they go through a
:class:`Backend`.  Two concrete backends ship:

* :class:`StatevectorBackend` — exact statevector simulation via
  ``qiskit.primitives.StatevectorSampler`` (pure Qiskit, no Aer/C++). Fast and
  deterministic under a seed; the default used by the test-suite.
* :class:`AerBackend` — high-performance shot-based simulation via
  ``qiskit_aer.primitives.SamplerV2``.  This is the "real quantum SDK" path and
  the one closest to how hardware behaves; swapping in an IBM Quantum runtime
  sampler here is all it takes to target real devices.

Bit convention (see :mod:`quantum_chaos.core.search_space`): factor ``i`` maps to
qubit ``i``.  Qiskit measurement keys are little-endian, so
:func:`~quantum_chaos.core.search_space.bitstring_to_config` reverses them; the
probability arrays returned here are indexed with factor ``0`` as the least
significant bit, matching
:func:`~quantum_chaos.core.search_space.config_to_index`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.search_space import Config, bitstring_to_config

try:  # pragma: no cover - import guard
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import Statevector

    _HAS_QISKIT = True
except Exception:  # pragma: no cover
    _HAS_QISKIT = False


class Backend:
    """Abstract execution backend."""

    name = "backend"

    def sample(self, circuit, shots: int) -> Dict[Config, int]:
        """Measure ``circuit`` ``shots`` times; return ``{config_tuple: count}``."""
        raise NotImplementedError

    def probabilities(self, circuit) -> np.ndarray:
        """Exact outcome probabilities, indexed with factor 0 as the LSB."""
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def _measured(circuit) -> "QuantumCircuit":
        meas = circuit.copy()
        meas.measure_all()
        return meas

    @staticmethod
    def _counts_to_configs(counts: Dict[str, int], num_factors: int) -> Dict[Config, int]:
        out: Dict[Config, int] = {}
        for key, cnt in counts.items():
            cfg = bitstring_to_config(key, num_factors)
            out[cfg] = out.get(cfg, 0) + int(cnt)
        return out


def _require_qiskit() -> None:
    if not _HAS_QISKIT:
        raise ImportError(
            "Qiskit is required for quantum backends. Install with:\n"
            "    pip install 'quantum-chaos[quantum]'"
        )


class StatevectorBackend(Backend):
    """Exact statevector simulation (pure Qiskit, no Aer)."""

    name = "statevector"

    def __init__(self, seed: Optional[int] = None):
        _require_qiskit()
        self.seed = seed
        self._sampler = StatevectorSampler(seed=seed)

    def sample(self, circuit, shots: int) -> Dict[Config, int]:
        n = circuit.num_qubits
        job = self._sampler.run([self._measured(circuit)], shots=shots)
        pub = job.result()[0]
        counts = pub.data.meas.get_counts()
        return self._counts_to_configs(counts, n)

    def probabilities(self, circuit) -> np.ndarray:
        # Statevector.probabilities() is indexed with qubit 0 as the LSB,
        # exactly our config-index convention.
        return np.asarray(Statevector(circuit).probabilities(), dtype=float)


class AerBackend(Backend):
    """Shot-based simulation via Qiskit Aer (the real-SDK path)."""

    name = "aer"

    def __init__(self, seed: Optional[int] = None):
        _require_qiskit()
        try:
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "qiskit-aer is required for the Aer backend. Install with:\n"
                "    pip install 'quantum-chaos[quantum]'"
            ) from exc
        self.seed = seed
        self._sampler = AerSamplerV2(seed=int(seed) if seed is not None else None)

    def sample(self, circuit, shots: int) -> Dict[Config, int]:
        n = circuit.num_qubits
        job = self._sampler.run([self._measured(circuit)], shots=shots)
        pub = job.result()[0]
        counts = pub.data.meas.get_counts()
        return self._counts_to_configs(counts, n)

    def probabilities(self, circuit) -> np.ndarray:
        # Aer's statevector method also matches the qubit-0-as-LSB convention.
        _require_qiskit()
        return np.asarray(Statevector(circuit).probabilities(), dtype=float)


_BACKENDS = {
    "statevector": StatevectorBackend,
    "aer": AerBackend,
}


def get_backend(name: str = "statevector", seed: Optional[int] = None) -> Backend:
    """Factory: ``"statevector"`` (default, fast) or ``"aer"`` (real SDK)."""
    key = name.lower()
    if key not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; choose from {list(_BACKENDS)}")
    return _BACKENDS[key](seed=seed)


def sample_configs(
    backend: Backend, circuit, shots: int
) -> List[Tuple[Config, float]]:
    """Sample and return ``(config, empirical_probability)`` pairs."""
    counts = backend.sample(circuit, shots)
    total = sum(counts.values()) or 1
    return [(cfg, cnt / total) for cfg, cnt in counts.items()]
