"""The optimization problem: a target system wrapped in a search space.

A :class:`FailureProblem` is what every optimizer consumes.  It owns the
memoization cache and the evaluation history so that:

* repeated visits to the same corner are free (quantum samplers revisit a lot),
* every optimizer produces a comparable ``num_evaluations`` count,
* results can be reconstructed from ``problem.history``.

Beyond the basics it also handles the messy realities of stress-testing real
systems:

* **Feasibility constraints** — infeasible fault combinations are scored with a
  steep penalty (and never executed on the target), so optimizers stay inside
  the reachable region without any special-casing.
* **Noisy targets** — an AI system rarely fails deterministically.  ``repeats``
  + an ``aggregator`` (mean, p95, CVaR, …) turn a stochastic score into a stable,
  tail-aware estimate, and per-config variance is tracked.
* **Pathological targets** — non-finite scores and exceptions are handled by an
  explicit policy.  Notably, a target that *crashes* on an input can be treated
  as a catastrophic failure (``on_error="as_failure"``), which is often exactly
  what you want to discover.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .result import OptimizationResult
from .search_space import Config, ConfigLike, SearchSpace

Config_ = Config

#: score assigned to the "worst" infeasible config; scaled by violation count.
_INFEASIBLE_BASE = 1e9


def _percentile_aggregator(q: float) -> Callable[[List[float]], float]:
    def agg(vals: List[float]) -> float:
        return float(np.percentile(vals, q))

    return agg


def _cvar_aggregator(alpha: float = 0.2) -> Callable[[List[float]], float]:
    """Mean of the worst (highest) ``alpha`` fraction — upper-tail failure risk."""

    def agg(vals: List[float]) -> float:
        arr = np.sort(np.asarray(vals, dtype=float))[::-1]
        k = max(1, int(math.ceil(alpha * len(arr))))
        return float(arr[:k].mean())

    return agg


_AGGREGATORS: Dict[str, Callable[[List[float]], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "max": lambda v: float(np.max(v)),
    "min": lambda v: float(np.min(v)),
    "median": lambda v: float(np.median(v)),
    "p95": _percentile_aggregator(95),
    "p05": _percentile_aggregator(5),
    "cvar": _cvar_aggregator(0.2),
}


class FailureProblem:
    """Bundle a target callable with a :class:`SearchSpace`.

    Parameters
    ----------
    target:
        A :class:`~quantum_chaos.targets.base.TargetSystem` or any callable
        taking a ``{factor_name: 0/1}`` dict and returning a failure score
        (higher = worse).
    search_space:
        Optional explicit space.  If ``target`` is a ``TargetSystem`` it can
        supply its own space, so this may be omitted.
    constraints:
        Optional :class:`~quantum_chaos.core.constraints.ConstraintSet`.  When
        supplied, infeasible configurations are penalized and never executed.
    repeats:
        Evaluate a (noisy) target this many times per configuration.
    aggregator:
        How to reduce ``repeats`` samples: one of ``"mean"``, ``"max"``,
        ``"min"``, ``"median"``, ``"p95"``, ``"p05"``, ``"cvar"``, or a callable
        ``list[float] -> float``.
    on_error:
        Policy when the target raises: ``"raise"`` (default), ``"as_failure"``
        (a crash is a catastrophic failure, scored ``failure_score``), or
        ``"as_safe"`` (scored ``safe_score``, effectively ignored).
    on_nonfinite:
        Policy for NaN/±inf scores: ``"raise"`` (default) or ``"clip"`` (map to
        ``clip_range``; NaN is treated as ``safe_score``).
    failure_score / safe_score:
        Sentinel magnitudes for the error policies above.
    clip_range:
        ``(low, high)`` bounds used by ``on_nonfinite="clip"``.
    infeasible_penalty:
        Base magnitude of the infeasible-config penalty (default ``1e9``).
    """

    def __init__(
        self,
        target: Union["TargetSystemLike", Callable[[Dict[str, int]], float]],
        search_space: Optional[SearchSpace] = None,
        constraints=None,
        repeats: int = 1,
        aggregator: Union[str, Callable[[List[float]], float]] = "mean",
        on_error: str = "raise",
        on_nonfinite: str = "raise",
        failure_score: float = 1e6,
        safe_score: float = -1e9,
        clip_range: Tuple[float, float] = (-1e9, 1e9),
        infeasible_penalty: float = _INFEASIBLE_BASE,
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

        # Attach constraints so constraint-aware sampling is transparent.
        self.constraints = constraints
        self.search_space.constraints = constraints

        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.repeats = int(repeats)
        self._aggregator = self._resolve_aggregator(aggregator)
        self.aggregator_name = aggregator if isinstance(aggregator, str) else "custom"

        if on_error not in ("raise", "as_failure", "as_safe"):
            raise ValueError(f"invalid on_error policy: {on_error!r}")
        if on_nonfinite not in ("raise", "clip"):
            raise ValueError(f"invalid on_nonfinite policy: {on_nonfinite!r}")
        self.on_error = on_error
        self.on_nonfinite = on_nonfinite
        self.failure_score = float(failure_score)
        self.safe_score = float(safe_score)
        self.clip_range = (float(clip_range[0]), float(clip_range[1]))
        self.infeasible_penalty = float(infeasible_penalty)

        self._cache: Dict[Config_, float] = {}
        self._variance: Dict[Config_, float] = {}
        self._history: List[Tuple[Config_, float]] = []
        self._num_evaluations = 0  # distinct configs actually executed on target
        self._num_queries = 0  # total evaluate() calls, cache hits included
        self._num_infeasible = 0  # distinct infeasible configs encountered
        self._num_errors = 0  # target exceptions handled by policy

    @staticmethod
    def _resolve_aggregator(aggregator) -> Callable[[List[float]], float]:
        if callable(aggregator):
            return aggregator
        if aggregator not in _AGGREGATORS:
            raise ValueError(
                f"unknown aggregator {aggregator!r}; choose from {list(_AGGREGATORS)}"
            )
        return _AGGREGATORS[aggregator]

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
    def num_infeasible(self) -> int:
        return self._num_infeasible

    @property
    def num_errors(self) -> int:
        return self._num_errors

    @property
    def history(self) -> List[Tuple[Config_, float]]:
        return self._history

    # -- feasibility helpers ---------------------------------------------
    def is_feasible(self, value: ConfigLike) -> bool:
        if self.constraints is None:
            return True
        return self.constraints.is_feasible(self.search_space.as_config(value))

    def _infeasible_score(self, bits: Config_) -> float:
        n_viol = len(self.constraints.violations(bits))
        # More violations → worse (more negative), guiding search toward the
        # feasible boundary without ever running the target.
        return -self.infeasible_penalty * (1 + n_viol)

    def score_variance(self, value: ConfigLike) -> float:
        """Sample variance recorded for a configuration (0 if deterministic)."""
        bits = self.search_space.as_config(value)
        return self._variance.get(bits, 0.0)

    # -- evaluation -------------------------------------------------------
    def evaluate(self, value: ConfigLike) -> float:
        """Return the (aggregated) failure score of a configuration, memoized."""
        bits = self.search_space.as_config(value)
        self._num_queries += 1
        cached = self._cache.get(bits)
        if cached is not None:
            return cached

        if self.constraints is not None and not self.constraints.is_feasible(bits):
            score = self._infeasible_score(bits)
            self._cache[bits] = score
            self._num_infeasible += 1
            return score  # never executed on the target; not in history

        score = self._run_target(bits)
        self._cache[bits] = score
        self._history.append((bits, score))
        self._num_evaluations += 1
        return score

    def _run_target(self, bits: Config_) -> float:
        config = {name: bit for name, bit in zip(self.search_space.names, bits)}
        samples: List[float] = []
        for _ in range(self.repeats):
            try:
                raw = float(self.target(config))
            except Exception:
                self._num_errors += 1
                if self.on_error == "raise":
                    raise
                raw = self.failure_score if self.on_error == "as_failure" else self.safe_score
            samples.append(self._sanitize(raw))
        if len(samples) > 1:
            self._variance[bits] = float(np.var(samples))
        return float(self._aggregator(samples))

    def _sanitize(self, value: float) -> float:
        if math.isfinite(value):
            return value
        if self.on_nonfinite == "raise":
            raise ValueError(f"target returned non-finite score: {value}")
        # clip policy
        if math.isnan(value):
            return self.safe_score  # a NaN failure is meaningless → treat as safe
        lo, hi = self.clip_range
        return hi if value > 0 else lo

    def reset_stats(self) -> None:
        """Clear cache/history so the problem can be reused for a fresh run."""
        self._cache.clear()
        self._variance.clear()
        self._history.clear()
        self._num_evaluations = 0
        self._num_queries = 0
        self._num_infeasible = 0
        self._num_errors = 0

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
            raise RuntimeError("no feasible evaluations were recorded")
        meta = dict(metadata or {})
        if self.constraints is not None:
            meta.setdefault("num_infeasible", self._num_infeasible)
            meta.setdefault("best_is_feasible", self.constraints.is_feasible(best_bits))
        if self._num_errors:
            meta.setdefault("num_errors", self._num_errors)
        if self.repeats > 1:
            meta.setdefault("repeats", self.repeats)
            meta.setdefault("aggregator", self.aggregator_name)
            meta.setdefault("best_variance", self._variance.get(best_bits, 0.0))
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
            metadata=meta,
        )


# Only used for type hints; kept as a soft reference to avoid import cycles.
TargetSystemLike = object
