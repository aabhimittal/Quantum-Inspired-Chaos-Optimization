"""Quantum-Inspired Chaos Optimization.

Use quantum heuristics to find worst-case failure combinations and stress-test
AI systems.  Classical chaos engineering throws random faults at a system;
this framework puts a quantum *superposition* over the whole space of failure
configurations, lets amplitude concentrate on the system's weak spots, and
collapses to reveal the exact combinations most likely to cause catastrophic
failure.

Quickstart
----------
>>> from quantum_chaos import FailureProblem, FunctionTarget
>>> from quantum_chaos.optimizers import VariationalQuantumSampler
>>> target = FunctionTarget(
...     lambda cfg: sum(cfg.values()),  # your AI system's failure score
...     num_factors=6, name="demo",
... )
>>> problem = FailureProblem(target)
>>> result = VariationalQuantumSampler(seed=0).optimize(problem)
>>> print(result.summary())  # doctest: +SKIP
"""

from ._version import __version__
from .core import (
    ChaosFactor,
    FailureProblem,
    OptimizationResult,
    SearchSpace,
)
from .targets import (
    FunctionTarget,
    TargetSystem,
    get_target,
    list_targets,
    register,
)

__all__ = [
    "__version__",
    "FailureProblem",
    "OptimizationResult",
    "SearchSpace",
    "ChaosFactor",
    "TargetSystem",
    "FunctionTarget",
    "get_target",
    "list_targets",
    "register",
]
