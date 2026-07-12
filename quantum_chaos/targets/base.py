"""Target systems — the things we are trying to break.

A *target* answers one question: given a failure configuration (which faults
are active), how badly does the system fail?  The score is a non-negative
number where **higher = worse**.  Everything downstream maximizes it.

Two ways to define a target:

* subclass :class:`TargetSystem` and implement :meth:`failure_score`, or
* wrap any Python callable with :class:`FunctionTarget`.

Optionally a target can advertise its known ``worst_config`` (ground truth),
which the test-suite and benchmarks use to check that an optimizer actually
found the darkest corner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

from ..core.search_space import ChaosFactor, SearchSpace


class TargetSystem(ABC):
    """Abstract base class for a stress-test target."""

    #: short, unique, human-readable identifier
    name: str = "target"
    #: number of binary chaos factors this target exposes
    num_factors: int = 0
    #: optional explicit factor names (defaults to ``f0..f{n-1}``)
    factor_names: Optional[Sequence[str]] = None

    @abstractmethod
    def failure_score(self, config: Dict[str, int]) -> float:
        """Return the failure severity for ``config`` (higher = worse)."""

    # Targets are callables so they slot straight into FailureProblem.
    def __call__(self, config: Dict[str, int]) -> float:
        return self.failure_score(config)

    def make_search_space(self) -> SearchSpace:
        if self.factor_names is not None:
            factors = [ChaosFactor(name=str(n)) for n in self.factor_names]
            return SearchSpace(factors)
        return SearchSpace(self.num_factors)

    def worst_config(self) -> Optional[Dict[str, int]]:
        """Ground-truth worst configuration, if known.  ``None`` otherwise."""
        return None

    def describe(self) -> str:
        return self.__doc__.strip().splitlines()[0] if self.__doc__ else self.name


class FunctionTarget(TargetSystem):
    """Wrap an arbitrary callable as a :class:`TargetSystem`.

    This is the generic escape hatch: point it at *any* function that scores
    how badly your real AI system fails under a given fault configuration.

    Parameters
    ----------
    fn:
        Callable.  It receives either a ``{name: 0/1}`` dict (default) or, if
        ``pass_dict=False``, a tuple of bits — and returns a failure score.
    num_factors:
        Number of binary chaos factors.
    name:
        Identifier for CLI / reporting.
    factor_names:
        Optional names for the factors.
    worst:
        Optional known-worst config (dict or bit sequence) for validation.
    pass_dict:
        If ``True`` (default) ``fn`` is called with a name->bit dict; otherwise
        with a plain bit tuple.
    description:
        One-line human description.
    """

    def __init__(
        self,
        fn: Callable,
        num_factors: int,
        name: str = "custom",
        factor_names: Optional[Sequence[str]] = None,
        worst=None,
        pass_dict: bool = True,
        description: str = "",
    ):
        self._fn = fn
        self.num_factors = int(num_factors)
        self.name = name
        self.factor_names = list(factor_names) if factor_names is not None else None
        self._pass_dict = pass_dict
        self._description = description or f"custom target '{name}'"
        self._worst = worst

    def failure_score(self, config: Dict[str, int]) -> float:
        if self._pass_dict:
            return float(self._fn(config))
        names = self.factor_names or [f"f{i}" for i in range(self.num_factors)]
        bits = tuple(int(config[n]) for n in names)
        return float(self._fn(bits))

    def worst_config(self) -> Optional[Dict[str, int]]:
        if self._worst is None:
            return None
        names = self.factor_names or [f"f{i}" for i in range(self.num_factors)]
        if isinstance(self._worst, dict):
            return {n: int(self._worst.get(n, 0)) for n in names}
        return {n: int(b) for n, b in zip(names, self._worst)}

    def describe(self) -> str:
        return self._description
