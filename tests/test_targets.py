import numpy as np
import pytest

from quantum_chaos import FailureProblem, FunctionTarget
from quantum_chaos.targets import (
    CorrelatedFaults,
    DeceptiveTrap,
    MaxSatFailure,
    NeedleInHaystack,
    SumThreshold,
    get_target,
    list_targets,
)

ALL = [
    DeceptiveTrap(num_factors=8, k=4),
    NeedleInHaystack(num_factors=8, seed=1),
    CorrelatedFaults(num_factors=8, seed=1),
    SumThreshold(num_factors=8),
    MaxSatFailure(num_factors=8, seed=1),
]


@pytest.mark.parametrize("target", ALL, ids=lambda t: t.name)
def test_worst_config_is_actually_worst(target):
    """The declared worst config must be the global maximum by brute force."""
    space = target.make_search_space()
    worst = target.worst_config()
    if worst is None:
        pytest.skip("no ground truth advertised")
    worst_bits = tuple(worst[n] for n in space.names)
    best_score = -np.inf
    best_bits = None
    for bits in space.all_configs():
        cfg = dict(zip(space.names, bits))
        s = target.failure_score(cfg)
        if s > best_score:
            best_score, best_bits = s, bits
    assert best_bits == worst_bits


def test_registry_lists_and_builds():
    assert set(list_targets()) >= {
        "deceptive_trap", "needle", "correlated_faults", "sum_threshold", "maxsat"
    }
    t = get_target("needle", num_factors=6)
    assert t.num_factors == 6


def test_function_target_dict_and_tuple_modes():
    f_dict = FunctionTarget(lambda c: sum(c.values()), num_factors=3, name="d")
    assert f_dict.failure_score({"f0": 1, "f1": 1, "f2": 0}) == 2

    f_tup = FunctionTarget(lambda b: sum(b), num_factors=3, name="t", pass_dict=False)
    assert f_tup.failure_score({"f0": 1, "f1": 0, "f2": 1}) == 2


def test_function_target_worst_config_roundtrip():
    t = FunctionTarget(
        lambda c: 1.0, num_factors=3, name="w",
        factor_names=["a", "b", "c"], worst=[1, 0, 1],
    )
    assert t.worst_config() == {"a": 1, "b": 0, "c": 1}


def test_problem_caches_and_counts_queries():
    t = CorrelatedFaults(num_factors=6, seed=0)
    p = FailureProblem(t)
    p.evaluate("101010")
    p.evaluate("101010")  # cache hit
    assert p.num_evaluations == 1  # distinct
    assert p.num_queries == 2  # every call counts
