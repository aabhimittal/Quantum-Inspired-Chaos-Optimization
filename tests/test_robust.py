"""Robust / noisy evaluation and hardening against pathological targets."""

import math

import numpy as np
import pytest

from quantum_chaos import FailureProblem, FunctionTarget


def _target(fn, n=4, name="t"):
    return FunctionTarget(fn, num_factors=n, name=name)


def test_repeats_and_variance_tracked():
    rng = np.random.default_rng(0)
    p = FailureProblem(_target(lambda c: sum(c.values()) + rng.normal(0, 1.0)),
                       repeats=16, aggregator="mean")
    val = p.evaluate("1111")
    assert p.score_variance("1111") > 0
    assert 2.0 < val < 6.0  # mean around 4


def test_aggregators_ordering():
    rng = np.random.default_rng(1)
    base = _target(lambda c: sum(c.values()) + rng.normal(0, 1.0))
    results = {}
    for agg in ["min", "mean", "p95", "max", "cvar"]:
        rng2 = np.random.default_rng(1)  # same noise stream per aggregator
        p = FailureProblem(
            _target(lambda c: sum(c.values()) + rng2.normal(0, 1.0)),
            repeats=64, aggregator=agg,
        )
        results[agg] = p.evaluate("1111")
    assert results["min"] <= results["mean"] <= results["max"]
    assert results["cvar"] >= results["mean"]  # upper-tail focus


def test_repeats_one_is_deterministic_passthrough():
    p = FailureProblem(_target(lambda c: 2.0 * sum(c.values())), repeats=1)
    assert p.evaluate("1010") == 4.0
    assert p.score_variance("1010") == 0.0


def test_custom_aggregator_callable():
    p = FailureProblem(_target(lambda c: sum(c.values())),
                       repeats=3, aggregator=lambda vals: sum(vals) + 100)
    assert p.evaluate("1100") == 2 + 2 + 2 + 100
    assert p.aggregator_name == "custom"


def test_on_error_raise_is_default():
    def boom(c):
        raise RuntimeError("crash")

    p = FailureProblem(_target(boom))
    with pytest.raises(RuntimeError):
        p.evaluate("1000")


def test_on_error_as_failure_treats_crash_as_catastrophe():
    def boom(c):
        if c["f0"]:
            raise RuntimeError("crash")
        return 1.0

    p = FailureProblem(_target(boom), on_error="as_failure", failure_score=500.0)
    assert p.evaluate("1000") == 500.0
    assert p.evaluate("0000") == 1.0
    assert p.num_errors == 1


def test_on_error_as_safe_ignores_crash():
    def boom(c):
        raise RuntimeError("crash")

    p = FailureProblem(_target(boom), on_error="as_safe", safe_score=-42.0)
    assert p.evaluate("1000") == -42.0


def test_on_nonfinite_raise_by_default():
    p = FailureProblem(_target(lambda c: float("inf")))
    with pytest.raises(ValueError):
        p.evaluate("1000")


def test_on_nonfinite_clip():
    p = FailureProblem(_target(lambda c: float("inf") if c["f0"] else float("-inf")),
                       on_nonfinite="clip", clip_range=(-7.0, 7.0))
    assert p.evaluate("1000") == 7.0
    assert p.evaluate("0100") == -7.0


def test_nan_clipped_to_safe():
    p = FailureProblem(_target(lambda c: float("nan")),
                       on_nonfinite="clip", safe_score=-1.5)
    assert p.evaluate("1000") == -1.5


def test_invalid_policies_rejected():
    with pytest.raises(ValueError):
        FailureProblem(_target(lambda c: 1.0), on_error="nope")
    with pytest.raises(ValueError):
        FailureProblem(_target(lambda c: 1.0), on_nonfinite="nope")
    with pytest.raises(ValueError):
        FailureProblem(_target(lambda c: 1.0), aggregator="nope")
    with pytest.raises(ValueError):
        FailureProblem(_target(lambda c: 1.0), repeats=0)
