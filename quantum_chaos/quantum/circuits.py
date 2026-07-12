"""Parameterized circuit builders (ansätze).

Two families:

* :func:`variational_ansatz` — a hardware-efficient RY + CX ansatz used by the
  Variational Quantum Sampler.  Its parameters define a *probability wave* over
  failure configurations; training reshapes that wave to pile amplitude onto the
  worst corners.
* :func:`qaoa_ansatz` — the standard QAOA alternating cost/mixer circuit built
  directly from an Ising Hamiltonian (see :mod:`quantum_chaos.quantum.qubo`).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - import guard
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    _HAS_QISKIT = True
except Exception:  # pragma: no cover
    _HAS_QISKIT = False


def _require_qiskit() -> None:
    if not _HAS_QISKIT:
        raise ImportError(
            "Qiskit is required to build circuits. Install with:\n"
            "    pip install 'quantum-chaos[quantum]'"
        )


def variational_ansatz(
    num_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
) -> Tuple["QuantumCircuit", "ParameterVector"]:
    """A hardware-efficient RY/CX ansatz.

    Parameters
    ----------
    num_qubits:
        One qubit per chaos factor.
    reps:
        Number of RY+entangler layers (expressibility vs. cost).
    entanglement:
        ``"linear"`` (CX chain) or ``"circular"`` (chain + wrap-around).

    Returns
    -------
    (circuit, params):
        An unmeasured circuit and its parameter vector of length
        ``(reps + 1) * num_qubits``.
    """
    _require_qiskit()
    num_params = (reps + 1) * num_qubits
    params = ParameterVector("θ", num_params)
    qc = QuantumCircuit(num_qubits)
    idx = 0
    for layer in range(reps):
        for q in range(num_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        if entanglement == "circular" and num_qubits > 2:
            qc.cx(num_qubits - 1, 0)
    for q in range(num_qubits):
        qc.ry(params[idx], q)
        idx += 1
    return qc, params


def qaoa_ansatz(
    linear: Dict[int, float],
    quadratic: Dict[Tuple[int, int], float],
    num_qubits: int,
    reps: int = 1,
) -> Tuple["QuantumCircuit", "ParameterVector", "ParameterVector"]:
    """Standard QAOA circuit for an Ising cost Hamiltonian.

    The Hamiltonian is ``sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j`` with
    ``linear = {i: h_i}`` and ``quadratic = {(i, j): J_ij}``.

    Returns ``(circuit, gammas, betas)`` where ``gammas``/``betas`` are length-
    ``reps`` parameter vectors.
    """
    _require_qiskit()
    gammas = ParameterVector("γ", reps)
    betas = ParameterVector("β", reps)
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))  # uniform superposition
    for p in range(reps):
        gamma = gammas[p]
        # Cost unitary exp(-i * gamma * H_C).
        for i, h in linear.items():
            if h != 0.0:
                qc.rz(2.0 * gamma * h, i)
        for (i, j), J in quadratic.items():
            if J != 0.0:
                qc.rzz(2.0 * gamma * J, i, j)
        # Mixer unitary exp(-i * beta * sum X).
        beta = betas[p]
        for q in range(num_qubits):
            qc.rx(2.0 * beta, q)
    return qc, gammas, betas


def bind(circuit, params, values):
    """Bind a parameter vector (or list) to numeric ``values``.

    Works across Qiskit 1.x/2.x where ``assign_parameters`` is the stable API.
    """
    mapping = {p: float(v) for p, v in zip(params, values)}
    return circuit.assign_parameters(mapping)


def bind_many(circuit, param_value_pairs):
    """Bind several (parameter_vector, values) pairs at once."""
    mapping = {}
    for params, values in param_value_pairs:
        for p, v in zip(params, values):
            mapping[p] = float(v)
    return circuit.assign_parameters(mapping)
