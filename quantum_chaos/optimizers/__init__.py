"""Failure-mode optimizers: quantum-inspired and classical baselines."""

from typing import Callable, Dict, List

from .annealing import SimulatedQuantumAnnealer
from .base import Optimizer
from .classical import HillClimb, RandomSearch
from .grover import GroverSearch
from .qaoa import QAOAOptimizer
from .vqs import VariationalQuantumSampler

#: name -> optimizer class, for the CLI / benchmarks
OPTIMIZERS: Dict[str, Callable[..., object]] = {
    "vqs": VariationalQuantumSampler,
    "qaoa": QAOAOptimizer,
    "grover": GroverSearch,
    "annealing": SimulatedQuantumAnnealer,
    "random": RandomSearch,
    "hillclimb": HillClimb,
}


def get_optimizer(name: str, **kwargs):
    """Instantiate an optimizer by name."""
    key = name.lower()
    if key not in OPTIMIZERS:
        raise KeyError(
            f"unknown optimizer {name!r}; available: {', '.join(sorted(OPTIMIZERS))}"
        )
    return OPTIMIZERS[key](**kwargs)


def list_optimizers() -> List[str]:
    return sorted(OPTIMIZERS)


__all__ = [
    "Optimizer",
    "VariationalQuantumSampler",
    "QAOAOptimizer",
    "GroverSearch",
    "SimulatedQuantumAnnealer",
    "RandomSearch",
    "HillClimb",
    "OPTIMIZERS",
    "get_optimizer",
    "list_optimizers",
]
