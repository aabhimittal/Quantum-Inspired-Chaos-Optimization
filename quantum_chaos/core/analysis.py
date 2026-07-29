"""Root-cause analysis of a discovered failure.

Finding *a* worst-case configuration is only half the job.  An engineer then
asks: *which* of these faults actually matter, and what is the smallest set that
still breaks the system?  These functions answer that by probing the target
around a configuration of interest (typically the worst one an optimizer found).

All of them route through :meth:`FailureProblem.evaluate`, so they respect the
cache, the budget counters, and any feasibility constraints.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Config = Tuple[int, ...]


def _flip(bits: Sequence[int], i: int) -> Config:
    out = list(bits)
    out[i] ^= 1
    return tuple(out)


def marginal_importance(problem, bits: Sequence[int]) -> Dict[str, float]:
    """How much failure each factor contributes at ``bits``.

    For each factor we flip it and measure the drop in failure:
    ``importance[f] = score(bits) - score(bits with f toggled)``.  A large
    positive value means that factor is load-bearing for the failure; a negative
    value means toggling it would make things *worse* still.

    Infeasible neighbours (under the problem's constraints) are reported as
    ``nan`` — you cannot attribute importance to a move you are not allowed to make.
    """
    names = problem.search_space.names
    bits = problem.search_space.as_config(bits)
    base = problem.evaluate(bits)
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        neighbour = _flip(bits, i)
        if problem.constraints is not None and not problem.constraints.is_feasible(neighbour):
            out[name] = float("nan")
            continue
        out[name] = float(base - problem.evaluate(neighbour))
    return out


def minimal_critical_set(
    problem,
    bits: Sequence[int],
    retain: float = 0.9,
) -> Dict[str, object]:
    """Smallest subset of the *active* faults that reproduces the failure.

    Starting from the active factors in ``bits`` (all others off), greedily drop
    factors as long as the remaining configuration still scores at least
    ``retain`` × the original failure.  The survivors are the root cause — the
    faults you actually need for the catastrophe.

    Returns a dict with the reduced ``config``/``bits``, the ``factors`` names,
    the achieved ``score`` and the ``target`` level.
    """
    names = problem.search_space.names
    n = problem.num_factors
    bits = problem.search_space.as_config(bits)
    base = problem.evaluate(bits)
    target_level = retain * base if base >= 0 else base / max(retain, 1e-9)

    current = {i for i in range(n) if bits[i]}

    def config_of(active: set) -> Config:
        return tuple(1 if k in active else 0 for k in range(n))

    improved = True
    while improved:
        improved = False
        # Try to drop the currently least-important factor first for a tighter set.
        for i in sorted(current):
            trial = current - {i}
            cfg = config_of(trial)
            if problem.constraints is not None and not problem.constraints.is_feasible(cfg):
                continue
            if problem.evaluate(cfg) >= target_level:
                current = trial
                improved = True
                break

    reduced = config_of(current)
    return {
        "bits": reduced,
        "config": {name: bit for name, bit in zip(names, reduced)},
        "factors": [names[i] for i in sorted(current)],
        "score": problem.evaluate(reduced),
        "base_score": base,
        "retain": retain,
        "target_level": target_level,
    }


def pairwise_interactions(
    problem,
    factors: Optional[Sequence[int]] = None,
    baseline: Optional[Sequence[int]] = None,
    top: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Detect non-additive interactions between factor pairs.

    For each pair ``(i, j)`` we measure the second-order finite difference around
    a ``baseline`` (all-off by default)::

        interaction = f(i,j on) - f(i on) - f(j on) + f(neither)

    A large positive value means the two faults are *synergistic* — together they
    break far more than the sum of their individual effects.  That is precisely
    the combination-driven failure this whole framework hunts for.

    Returns ``(name_i, name_j, interaction)`` sorted by descending magnitude.
    """
    names = problem.search_space.names
    n = problem.num_factors
    idxs = list(factors) if factors is not None else list(range(n))
    base = list(problem.search_space.as_config(baseline)) if baseline is not None else [0] * n

    def with_on(*on: int) -> Config:
        cfg = list(base)
        for k in on:
            cfg[k] = 1
        return tuple(cfg)

    def safe_eval(cfg: Config) -> Optional[float]:
        if problem.constraints is not None and not problem.constraints.is_feasible(cfg):
            return None
        return problem.evaluate(cfg)

    results: List[Tuple[str, str, float]] = []
    f_neither = safe_eval(tuple(base))
    if f_neither is None:
        return results
    for i, j in combinations(idxs, 2):
        f_i = safe_eval(with_on(i))
        f_j = safe_eval(with_on(j))
        f_ij = safe_eval(with_on(i, j))
        if None in (f_i, f_j, f_ij):
            continue
        interaction = f_ij - f_i - f_j + f_neither
        results.append((names[i], names[j], float(interaction)))

    results.sort(key=lambda t: -abs(t[2]))
    if top is not None:
        results = results[:top]
    return results
