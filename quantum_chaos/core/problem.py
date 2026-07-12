"""The optimization problem: a target system wrapped in a search space.

A :class:`FailureProblem` is what every optimizer consumes.  It owns the
memoization cache and the evaluation history so that:

* repeated visits to the same corner are free (quantum samplers revisit a lot),
* every optimizer produces a comparable ``num_evaluations`` count,
* results can be reconstructed from ``problem.history``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

from .result import OptimizationResult
from .search_space import Config, ConfigLike, SearchSpace

Config_ = Config


class FailureProblem:
    """Bundle a target callable with a :class:`SearchSpace`.

    Parameters
    ----------
    target:
        Either a :class:`~quantum_chaos.targets.base.TargetSystem` or any
        callable taking a ``{factor_name: 0/1}`` dict and returning a failure
        score (higher = worse).
    search_space:
        Optional explicit space.  If ``target`` is a ``TargetSystem`` it can
        supply its own space, so this may be omitted.
    """

    def __init__(
        self,
        target: Union["TargetSystemLike", Callable[[Dict[str, int]], float]],
        search_space: Optional[SearchSpace] = None,
    ):
        # Late import avoided: duck-type the TargetSystem interface.
        space = search_space
        if space is None:
            make_space = getattr(target, "make_search_space", None)
            if callable(make_space):
                space = make_space()
        if space is None:
            raise ValueError(
                "search_space is required unless target provides make_search_space()"
            )
        self.search_space = space
        self.target = target
        self.name = getattr(target, "name", getattr(target, "__name__", "target"))

        self._cache: Dict[Config_, float] = {}
        self._history: List[Tuple[Config_, float]] = []
        self._num_evaluations = 0  # distinct configs actually executed
        self._num_queries = 0  # total evaluate() calls, cache hits included

    # -- properties -------------------------------------------------------
    @property
    def num_qubits(self) -> int:
        return self.search_space.num_factors

    @property
    def num_factors(self) -> int:
        return self.search_space.num_factors

    @property
    def num_evaluations(self) -> int:
        """Distinct configurations actually executed on the target."""
        return self._num_evaluations

    @property
    def num_queries(self) -> int:
        """Total ``evaluate`` calls issued, including cache hits.

        This is the budget currency: it advances on every request even when the
        answer is served from cache, so budget-bounded local searches always
        terminate.  ``num_evaluations`` is the efficiency metric (real work).
        """
        return self._num_queries

    @property
    def history(self) -> List[Tuple[Config_, float]]:
        return self._history

    # -- evaluation -------------------------------------------------------
    def evaluate(self, value: ConfigLike) -> float:
        """Return the failure score of a configuration (memoized).

        Only cache-*misses* increment :attr:`num_evaluations`, so the count
        reflects genuine work done on the target — the fair currency for
        benchmarking one optimizer against another.
        """
        bits = self.search_space.as_config(value)
        self._num_queries += 1
        cached = self._cache.get(bits)
        if cached is not None:
            return cached
        config = {name: bit for name, bit in zip(self.search_space.names, bits)}
        score = float(self.target(config))
        self._cache[bits] = score
        self._history.append((bits, score))
        self._num_evaluations += 1
        return score

    def reset_stats(self) -> None:
        """Clear cache/history so the problem can be reused for a fresh run."""
        self._cache.clear()
        self._history.clear()
        self._num_evaluations = 0
        self._num_queries = 0

    # -- helpers ----------------------------------------------------------
    def best_seen(self) -> Tuple[Optional[Config_], float]:
        best_bits: Optional[Config_] = None
        best_score = float("-inf")
        for bits, score in self._history:
            if score > best_score:
                best_score, best_bits = score, bits
        return best_bits, best_score

    def build_result(
        self,
        optimizer: str,
        backend: str = "",
        metadata: Optional[Dict] = None,
        best_bits: Optional[Config_] = None,
    ) -> OptimizationResult:
        """Assemble an :class:`OptimizationResult` from the recorded history."""
        if best_bits is None:
            best_bits, best_score = self.best_seen()
        else:
            best_score = self._cache[best_bits]
        if best_bits is None:
            raise RuntimeError("no evaluations were recorded")
        return OptimizationResult(
            best_bits=best_bits,
            best_score=best_score,
            best_config={
                name: bit for name, bit in zip(self.search_space.names, best_bits)
            },
            history=list(self._history),
            num_evaluations=self._num_evaluations,
            num_queries=self._num_queries,
            optimizer=optimizer,
            backend=backend,
            metadata=metadata or {},
        )


# Only used for type hints; kept as a soft reference to avoid import cycles.
TargetSystemLike = object
