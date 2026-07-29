"""Industrial edge cases: degenerate spaces, pathological targets, tight limits."""

import numpy as np
import pytest

from quantum_chaos import ConstraintSet, FailureProblem, FunctionTarget, SearchSpace, get_target
from quantum_chaos.optimizers import (
    HillClimb,
    RandomSearch,
    SimulatedQuantumAnnealer,
    VariationalQuantumSampler,
)


def test_single_factor_space():
    t = FunctionTarget(lambda c: float(c["f0"]), num_factors=1, name="one")
    p = FailureProblem(t)
    r = VariationalQuantumSampler(seed=0, iterations=15, shots=64,
                                  max_evaluations=500).optimize(p)
    assert r.best_bits == (1,)
    assert r.best_score == 1.0


def test_flat_landscape_terminates():
    t = FunctionTarget(lambda c: 3.14, num_factors=6, name="flat")
    for opt in [
        RandomSearch(seed=0, max_evaluations=300),
        SimulatedQuantumAnnealer(seed=0, max_evaluations=300),
        VariationalQuantumSampler(seed=0, iterations=10, shots=64, max_evaluations=500),
    ]:
        p = FailureProblem(t)
        r = opt.optimize(p)
        assert r.best_score == 3.14
        # Terminates within budget; sampling optimizers may overshoot by up to
        # one measurement round (budget is checked between iterations).
        overshoot = 4 * getattr(opt, "shots", 1)
        assert r.num_queries <= opt.max_evaluations + overshoot


def test_tiny_budget_still_returns():
    t = get_target("sum_threshold", num_factors=6)
    p = FailureProblem(t)
    r = RandomSearch(seed=0, max_evaluations=1).optimize(p)
    assert r.best_bits is not None
    assert r.num_queries <= 2


def test_negative_scores_are_maximized():
    # worst (max) failure of -sum is at the all-zeros corner (score 0).
    t = FunctionTarget(lambda c: -float(sum(c.values())), num_factors=6, name="neg")
    p = FailureProblem(t)
    r = HillClimb(seed=0, max_evaluations=1500).optimize(p)
    assert r.best_bits == (0,) * 6
    assert r.best_score == 0.0


def test_int_and_bool_return_coerced():
    t = FunctionTarget(lambda c: sum(c.values()) > 2, num_factors=4, name="boolret")
    p = FailureProblem(t)
    v = p.evaluate("1110")  # sum 3 > 2 -> True -> 1.0
    assert isinstance(v, float) and v == 1.0


def test_max_active_zero_only_empty_feasible():
    t = get_target("sum_threshold", num_factors=5)
    c = ConstraintSet.build(t.make_search_space(), max_active=0)
    p = FailureProblem(t, constraints=c)
    r = RandomSearch(seed=0, max_evaluations=500).optimize(p)
    assert r.best_bits == (0,) * 5


def test_contradictory_constraints_raise_cleanly():
    # required 3 factors but allow at most 1 active -> nothing feasible.
    t = get_target("sum_threshold", num_factors=4)
    c = ConstraintSet.build(t.make_search_space(), max_active=1,
                            required=["f0", "f1", "f2"])
    p = FailureProblem(t, constraints=c)
    with pytest.raises(RuntimeError):
        RandomSearch(seed=0, max_evaluations=300).optimize(p)


def test_large_space_feasible_sampling_is_fast():
    c = ConstraintSet(num_factors=25, max_active=2)
    rng = np.random.default_rng(0)
    for _ in range(30):
        bits = c.feasible_random(rng)
        assert c.is_feasible(bits)
        assert sum(bits) <= 2


def test_reset_stats_clears_all_counters():
    t = get_target("correlated_faults", num_factors=6)
    c = ConstraintSet.build(t.make_search_space(), max_active=3)
    p = FailureProblem(t, constraints=c)
    p.evaluate((1, 1, 1, 1, 1, 1))  # infeasible
    p.evaluate((1, 1, 0, 0, 0, 0))  # feasible
    assert p.num_infeasible == 1 and p.num_evaluations == 1
    p.reset_stats()
    assert p.num_infeasible == 0
    assert p.num_evaluations == 0
    assert p.num_queries == 0


def test_cascading_failure_target_behaviour():
    t = get_target("cascading_failure", num_factors=12, seed=17)
    p = FailureProblem(t)
    none_down = p.evaluate((0,) * 12)
    some_down = p.evaluate((1,) * 12)
    assert none_down == 0.0
    assert some_down == 12.0  # injecting everything downs everything
    assert t.worst_config() is None


def test_constrained_recovery_within_blast_radius():
    # With a blast radius of 2 the reachable worst is a single interacting pair.
    t = get_target("correlated_faults", num_factors=8, seed=1)
    c = ConstraintSet.build(t.make_search_space(), max_active=2)
    p = FailureProblem(t, constraints=c)
    r = RandomSearch(seed=0, max_evaluations=4000).optimize(p)
    assert sum(r.best_bits) <= 2
    # the best feasible pair should activate one of the ground-truth pairs
    active = {i for i, b in enumerate(r.best_bits) if b}
    assert any(set(pair) == active for pair in t._pairs)
