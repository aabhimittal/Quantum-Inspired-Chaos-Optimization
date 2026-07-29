"""Target systems and the built-in synthetic failure landscapes."""

from .base import FunctionTarget, TargetSystem
from .functions import (
    CascadingFailure,
    CorrelatedFaults,
    DeceptiveTrap,
    MaxSatFailure,
    NeedleInHaystack,
    SumThreshold,
)
from .registry import (
    get_target,
    list_targets,
    register,
    target_descriptions,
)

__all__ = [
    "TargetSystem",
    "FunctionTarget",
    "DeceptiveTrap",
    "NeedleInHaystack",
    "CorrelatedFaults",
    "SumThreshold",
    "MaxSatFailure",
    "CascadingFailure",
    "get_target",
    "list_targets",
    "register",
    "target_descriptions",
]
