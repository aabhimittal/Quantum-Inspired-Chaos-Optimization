import numpy as np
import pytest

from quantum_chaos import ConstraintSet, FailureProblem, SearchSpace, get_target
from quantum_chaos.optimizers import RandomSearch, VariationalQuantumSampler


def test_cardinality_feasibility_and_violations():
    c = ConstraintSet(num_factors=5, max_active=2)
    assert c.is_feasible((1, 1, 0, 0, 0))
    assert not c.is_feasible((1, 1, 1, 0, 0))
    assert any("max_active" in v for v in c.violations((1, 1, 1, 0, 0)))


def test_min_active():
    c = ConstraintSet(num_factors=4, min_active=2)
    assert not c.is_feasible((1, 0, 0, 0))
    assert c.is_feasible((1, 1, 0, 0))


def test_forbidden_pairs_and_mutual_exclusion():
    c = ConstraintSet(num_factors=4, forbidden_pairs=[(0, 1)],
                      mutual_exclusions=[[1, 2, 3]])
    assert not c.is_feasible((1, 1, 0, 0))  # forbidden pair
    assert not c.is_feasible((0, 1, 1, 0))  # >1 in exclusion group
    assert c.is_feasible((1, 0, 1, 0))


def test_implications_and_required():
    c = ConstraintSet(num_factors=3, implications=[(0, 1)], required=[2])
    assert not c.is_feasible((1, 0, 1))  # 0 active but 1 not
    assert not c.is_feasible((0, 0, 0))  # 2 required
    assert c.is_feasible((1, 1, 1))


def test_weights_and_budget():
    c = ConstraintSet(num_factors=3, weights=[1.0, 5.0, 2.0], budget=3.0)
    assert c.is_feasible((1, 0, 1))  # cost 3.0
    assert not c.is_feasible((0, 1, 0))  # cost 5.0 > 3
    assert c.cost((1, 0, 1)) == 3.0


def test_repair_yields_feasible():
    rng = np.random.default_rng(0)
    c = ConstraintSet(num_factors=8, max_active=3, forbidden_pairs=[(0, 1)],
                      required=[7], implications=[(2, 3)])
    for _ in range(50):
        bits = tuple(int(x) for x in rng.integers(0, 2, size=8))
        repaired = c.repair(bits, rng)
        assert c.is_feasible(repaired), (bits, repaired, c.violations(repaired))


def test_feasible_random_always_feasible():
    rng = np.random.default_rng(1)
    c = ConstraintSet(num_factors=10, max_active=3)
    for _ in range(100):
        assert c.is_feasible(c.feasible_random(rng))


def test_feasible_fraction_exact_small():
    c = ConstraintSet(num_factors=4, max_active=1)
    # feasible = configs with <=1 active = 1 + 4 = 5 out of 16
    assert c.feasible_fraction() == pytest.approx(5 / 16)


def test_build_from_names():
    space = SearchSpace(["a", "b", "c"])
    c = ConstraintSet.build(space, max_active=1, forbidden_pairs=[("a", "b")],
                            weights={"c": 2.0}, budget=2.0)
    assert c.forbidden_pairs == [(0, 1)]
    assert c.weights[2] == 2.0
    with pytest.raises(KeyError):
        ConstraintSet.build(space, forbidden_pairs=[("a", "zzz")])


def test_predicate_constraint():
    # custom: parity must be even
    c = ConstraintSet(num_factors=4, predicates=[lambda b: sum(b) % 2 == 0],
                      predicate_names=["even_parity"])
    assert c.is_feasible((1, 1, 0, 0))
    assert not c.is_feasible((1, 0, 0, 0))
    assert "even_parity" in " ".join(c.violations((1, 0, 0, 0)))


def test_broken_predicate_is_violation_not_crash():
    def boom(bits):
        raise RuntimeError("bad predicate")

    c = ConstraintSet(num_factors=3, predicates=[boom])
    assert not c.is_feasible((0, 0, 0))  # violation recorded, no exception


def test_search_space_samples_feasibly_when_constrained():
    space = SearchSpace(6)
    space.constraints = ConstraintSet(num_factors=6, max_active=2)
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert sum(space.random_config(rng)) <= 2


def test_problem_penalizes_infeasible_without_running_target():
    calls = {"n": 0}

    def target(cfg):
        calls["n"] += 1
        return float(sum(cfg.values()))

    from quantum_chaos import FunctionTarget

    t = FunctionTarget(target, num_factors=5, name="counted")
    c = ConstraintSet(num_factors=5, max_active=2)
    p = FailureProblem(t, constraints=c)
    infeasible_score = p.evaluate((1, 1, 1, 1, 1))  # 5 active > 2
    assert infeasible_score < 0  # steep penalty
    assert calls["n"] == 0  # target never ran on infeasible config
    assert p.num_infeasible == 1
    assert p.num_evaluations == 0


def test_optimizer_respects_constraints():
    t = get_target("correlated_faults", num_factors=10)
    c = ConstraintSet.build(t.make_search_space(), max_active=3)
    p = FailureProblem(t, constraints=c)
    r = RandomSearch(seed=0, max_evaluations=3000).optimize(p)
    assert sum(r.best_bits) <= 3
    assert c.is_feasible(r.best_bits)
    assert r.metadata["best_is_feasible"] is True
