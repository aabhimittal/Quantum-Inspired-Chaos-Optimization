"""Core data structures: search space, problem, result, constraints, analysis."""

from .analysis import marginal_importance, minimal_critical_set, pairwise_interactions
from .constraints import ConstraintSet
from .problem import FailureProblem
from .report import StressReport
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
    "ConstraintSet",
    "StressReport",
    "ChaosFactor",
    "SearchSpace",
    "marginal_importance",
    "minimal_critical_set",
    "pairwise_interactions",
    "bitstring_to_config",
    "config_to_index",
    "index_to_config",
]
