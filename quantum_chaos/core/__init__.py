"""Core data structures: search space, problem, result."""

from .problem import FailureProblem
from .result import OptimizationResult
from .search_space import (
    ChaosFactor,
    SearchSpace,
    bitstring_to_config,
    config_to_index,
    index_to_config,
)

__all__ = [
    "FailureProblem",
    "OptimizationResult",
    "ChaosFactor",
    "SearchSpace",
    "bitstring_to_config",
    "config_to_index",
    "index_to_config",
]
