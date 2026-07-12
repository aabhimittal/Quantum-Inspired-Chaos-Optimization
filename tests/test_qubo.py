import numpy as np
import pytest

from quantum_chaos import FailureProblem
from quantum_chaos.quantum.qubo import QUBO, fit_qubo_surrogate, ising_energy
from quantum_chaos.targets import CorrelatedFaults


def test_qubo_energy_matches_definition():
    q = QUBO(num_vars=2, linear=np.array([1.0, 2.0]), quadratic={(0, 1): 3.0}, offset=0.5)
    # f(1,1) = 0.5 + 1 + 2 + 3
    assert q.energy([1, 1]) == 6.5
    assert q.energy([0, 0]) == 0.5


def test_to_ising_ground_state_is_max_failure():
    """The Ising cost's minimum must sit at the QUBO's maximum-failure config."""
    q = QUBO(num_vars=3, linear=np.array([1.0, -0.5, 0.2]),
             quadratic={(0, 1): 2.0, (1, 2): -1.0}, offset=0.0)
    h, J, off = q.to_ising()

    best_fail = -np.inf
    best_bits = None
    min_cost = np.inf
    min_bits = None
    for i in range(8):
        bits = tuple((i >> k) & 1 for k in range(3))
        f = q.energy(bits)
        if f > best_fail:
            best_fail, best_bits = f, bits
        c = ising_energy(h, J, off, bits)
        if c < min_cost:
            min_cost, min_bits = c, bits
    assert best_bits == min_bits


def test_to_ising_energy_equals_negative_failure():
    q = QUBO(num_vars=3, linear=np.array([0.7, -0.3, 0.9]),
             quadratic={(0, 2): 1.5}, offset=0.0)
    h, J, off = q.to_ising()
    for i in range(8):
        bits = tuple((i >> k) & 1 for k in range(3))
        assert ising_energy(h, J, off, bits) == pytest.approx(-q.energy(bits))


def test_surrogate_fits_quadratic_target_well():
    # CorrelatedFaults is genuinely quadratic, so the surrogate should be exact.
    t = CorrelatedFaults(num_factors=6, seed=2)
    p = FailureProblem(t)
    rng = np.random.default_rng(0)
    # Tiny ridge for a near-exact fit of a genuinely quadratic function.
    q = fit_qubo_surrogate(p, num_samples=300, rng=rng, ridge=1e-8)
    space = t.make_search_space()
    errs = []
    for bits in space.all_configs():
        true = t.failure_score(dict(zip(space.names, bits)))
        errs.append(abs(true - q.energy(bits)))
    assert max(errs) < 1e-3
