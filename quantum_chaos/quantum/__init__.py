"""Quantum layer: backends, circuits, and QUBO surrogate machinery."""

from .backends import (
    AerBackend,
    Backend,
    StatevectorBackend,
    get_backend,
    sample_configs,
)
from .circuits import bind, bind_many, qaoa_ansatz, variational_ansatz
from .qubo import QUBO, fit_qubo_surrogate, ising_energy

__all__ = [
    "Backend",
    "StatevectorBackend",
    "AerBackend",
    "get_backend",
    "sample_configs",
    "variational_ansatz",
    "qaoa_ansatz",
    "bind",
    "bind_many",
    "QUBO",
    "fit_qubo_surrogate",
    "ising_energy",
]
