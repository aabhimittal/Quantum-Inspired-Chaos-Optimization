"""Each optimizer should recover the known worst config on an easy target.

Kept small and seeded so the suite runs in a few seconds on the statevector
backend (no Aer / no hardware).
"""

import numpy as np
import pytest

from quantum_chaos import FailureProblem
from quantum_chaos.optimizers import (
    GroverSearch,
    HillClimb,
    QAOAOptimizer,
    RandomSearch,
    SimulatedQuantumAnnealer,
    VariationalQuantumSampler,
)
from quantum_chaos.targets import CorrelatedFaults, NeedleInHaystack, SumThreshold


def _worst_bits(target):
    space = target.make_search_space()
    worst = target.worst_config()
    return tuple(worst[n] for n in space.names)


def test_vqs_finds_worst_correlated_faults():
    t = CorrelatedFaults(num_factors=6, seed=1)
    p = FailureProblem(t)
    r = VariationalQuantumSampler(reps=2, shots=256, iterations=40,
                                  seed=0, max_evaluations=4000).optimize(p)
    assert r.best_bits == _worst_bits(t)
    assert r.num_queries > 0
    assert r.metadata["final_cvar"] >= r.metadata["initial_cvar"]


def test_qaoa_finds_worst_on_quadratic_target():
    t = CorrelatedFaults(num_factors=6, seed=3)
    p = FailureProblem(t)
    r = QAOAOptimizer(reps=2, iterations=60, surrogate_samples=120, seed=0).optimize(p)
    assert r.best_bits == _worst_bits(t)


def test_grover_finds_the_needle():
    t = NeedleInHaystack(num_factors=8, seed=5)
    p = FailureProblem(t)
    r = GroverSearch(shots=512, seed=0).optimize(p)
    assert r.best_bits == _worst_bits(t)
    assert r.metadata["speedup_factor"] > 1.0


def test_annealing_and_baselines_on_monotone_target():
    # SumThreshold is monotone: worst = all ones; every method should find it.
    for opt in [
        SimulatedQuantumAnnealer(seed=0, max_evaluations=1500),
        RandomSearch(seed=0, max_evaluations=1500),
        HillClimb(seed=0, max_evaluations=1500),
    ]:
        t = SumThreshold(num_factors=8)
        p = FailureProblem(t)
        r = opt.optimize(p)
        assert r.best_bits == _worst_bits(t)


def test_budget_is_respected_and_terminates():
    t = CorrelatedFaults(num_factors=5, seed=0)
    p = FailureProblem(t)
    r = RandomSearch(seed=0, max_evaluations=200).optimize(p)
    # Small space (32 configs) fully cached; must still stop at the query budget.
    assert r.num_queries <= 200
    assert r.num_evaluations <= 32


def test_grover_rejects_too_many_qubits():
    t = SumThreshold(num_factors=20)
    p = FailureProblem(t)
    with pytest.raises(ValueError):
        GroverSearch(seed=0).optimize(p)
