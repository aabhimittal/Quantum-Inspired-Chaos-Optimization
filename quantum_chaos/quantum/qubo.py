"""QUBO surrogate models and QUBO -> Ising conversion.

QAOA needs the objective expressed as a quadratic function of binary variables
(a QUBO / Ising Hamiltonian).  A black-box failure function is not quadratic, so
we fit a **quadratic surrogate** from a handful of sampled evaluations, then let
QAOA optimize the surrogate.  We always re-score sampled bitstrings on the true
target, so the surrogate only *guides* the quantum search — it never replaces
the real failure signal.

Convention: the surrogate models *failure* (higher = worse).  Because QAOA
minimizes energy, :meth:`QUBO.to_ising` returns the Ising form of ``-failure``,
whose ground state is the worst-case configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class QUBO:
    """A quadratic model ``f(x) = offset + a·x + sum_{i<j} b_ij x_i x_j``.

    ``f`` estimates *failure severity* (higher = worse) over ``x in {0,1}^n``.
    """

    num_vars: int
    linear: np.ndarray  # shape (n,)
    quadratic: Dict[Tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0

    def energy(self, x: Sequence[int]) -> float:
        """Surrogate failure value of a configuration."""
        x = np.asarray(x, dtype=float)
        val = self.offset + float(self.linear @ x)
        for (i, j), b in self.quadratic.items():
            val += b * x[i] * x[j]
        return val

    def to_ising(self) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
        """Ising ``(h, J, offset)`` of the *cost to minimize* (``-failure``).

        Uses the standard substitution ``x_i = (1 - z_i) / 2``.  The returned
        Hamiltonian ``sum h_i z_i + sum J_ij z_i z_j + offset`` has its ground
        state at the maximum-failure configuration.
        """
        n = self.num_vars
        h = {i: 0.0 for i in range(n)}
        J: Dict[Tuple[int, int], float] = {}
        const = 0.0

        # Linear part of the failure model: sum a_i x_i.
        for i in range(n):
            a = float(self.linear[i])
            const += a / 2.0
            h[i] += -a / 2.0
        # Quadratic part: sum b_ij x_i x_j.
        for (i, j), b in self.quadratic.items():
            const += b / 4.0
            h[i] += -b / 4.0
            h[j] += -b / 4.0
            key = (min(i, j), max(i, j))
            J[key] = J.get(key, 0.0) + b / 4.0

        # Negate everything: cost = -failure (QAOA minimizes cost).
        h = {i: -v for i, v in h.items() if abs(v) > 1e-12}
        J = {k: -v for k, v in J.items() if abs(v) > 1e-12}
        const = -const
        return h, J, const


def fit_qubo_surrogate(
    problem,
    num_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
    ridge: float = 1e-3,
    max_order: int = 2,
) -> QUBO:
    """Fit a quadratic failure surrogate from random samples of ``problem``.

    Draws ``num_samples`` random configurations, evaluates the true target on
    each (these count toward the problem's evaluation budget), and solves a
    ridge-regularized least-squares fit over linear + pairwise features.

    Returns a :class:`QUBO` approximating the failure landscape.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = problem.num_factors
    space = problem.search_space

    pairs: List[Tuple[int, int]] = list(combinations(range(n), 2)) if max_order >= 2 else []
    num_features = 1 + n + len(pairs)

    X = np.zeros((num_samples, num_features))
    y = np.zeros(num_samples)
    for s in range(num_samples):
        bits = space.random_config(rng)
        y[s] = problem.evaluate(bits)
        row = [1.0]
        row.extend(float(b) for b in bits)
        for (i, j) in pairs:
            row.append(float(bits[i] * bits[j]))
        X[s] = row

    # Ridge regression: (X^T X + λI) w = X^T y  (do not penalize the bias).
    reg = ridge * np.eye(num_features)
    reg[0, 0] = 0.0
    w = np.linalg.solve(X.T @ X + reg, X.T @ y)

    offset = float(w[0])
    linear = w[1 : 1 + n].astype(float)
    quadratic: Dict[Tuple[int, int], float] = {}
    for idx, (i, j) in enumerate(pairs):
        coeff = float(w[1 + n + idx])
        if abs(coeff) > 1e-12:
            quadratic[(i, j)] = coeff
    return QUBO(num_vars=n, linear=linear, quadratic=quadratic, offset=offset)


def ising_energy(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    offset: float,
    bits: Sequence[int],
) -> float:
    """Energy of the Ising cost for a 0/1 configuration (``z = 1 - 2x``)."""
    z = {i: 1 - 2 * int(b) for i, b in enumerate(bits)}
    e = offset
    for i, hi in h.items():
        e += hi * z[i]
    for (i, j), Jij in J.items():
        e += Jij * z[i] * z[j]
    return float(e)
