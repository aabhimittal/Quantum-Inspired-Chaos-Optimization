"""A tiny name -> target-factory registry used by the CLI and examples."""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import TargetSystem
from .functions import (
    CorrelatedFaults,
    DeceptiveTrap,
    MaxSatFailure,
    NeedleInHaystack,
    SumThreshold,
)

#: name -> factory(**kwargs) -> TargetSystem
_REGISTRY: Dict[str, Callable[..., TargetSystem]] = {}


def register(name: str, factory: Callable[..., TargetSystem]) -> None:
    """Register a target factory under ``name`` (overwrites)."""
    _REGISTRY[name] = factory


def get_target(name: str, **kwargs) -> TargetSystem:
    """Instantiate a registered target by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown target {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


def list_targets() -> List[str]:
    return sorted(_REGISTRY)


def target_descriptions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, factory in sorted(_REGISTRY.items()):
        try:
            out[name] = factory().describe()
        except Exception:  # pragma: no cover - description best-effort
            out[name] = factory.__doc__ or name
    return out


# Register the built-in synthetic failure landscapes.
register("deceptive_trap", DeceptiveTrap)
register("needle", NeedleInHaystack)
register("correlated_faults", CorrelatedFaults)
register("sum_threshold", SumThreshold)
register("maxsat", MaxSatFailure)
