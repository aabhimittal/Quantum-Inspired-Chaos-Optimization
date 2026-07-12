from quantum_chaos.core.result import OptimizationResult


def _sample_result():
    history = [((0, 0), 0.1), ((1, 0), 0.4), ((0, 1), 0.2), ((1, 1), 0.9)]
    return OptimizationResult(
        best_bits=(1, 1),
        best_score=0.9,
        best_config={"a": 1, "b": 1},
        history=history,
        num_evaluations=4,
        num_queries=6,
        optimizer="vqs",
        backend="statevector",
        metadata={"reps": 2},
    )


def test_convergence_is_monotone_best_so_far():
    r = _sample_result()
    assert r.convergence() == [0.1, 0.4, 0.4, 0.9]


def test_best_string_and_active_factors():
    r = _sample_result()
    assert r.best_string == "11"
    assert r.active_factors == ["a", "b"]


def test_to_dict_roundtrip_fields():
    r = _sample_result()
    d = r.to_dict()
    assert d["best_string"] == "11"
    assert d["num_evaluations"] == 4
    assert d["num_queries"] == 6
    assert d["active_factors"] == ["a", "b"]
    assert d["metadata"]["reps"] == 2


def test_summary_contains_key_lines():
    r = _sample_result()
    s = r.summary()
    assert "worst combo" in s
    assert "vqs" in s
