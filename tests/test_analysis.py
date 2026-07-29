import math

import numpy as np

from quantum_chaos import (
    ConstraintSet,
    FailureProblem,
    StressReport,
    get_target,
    marginal_importance,
    minimal_critical_set,
    pairwise_interactions,
)
from quantum_chaos.targets import CorrelatedFaults


def _problem(seed=1, n=8):
    t = CorrelatedFaults(num_factors=n, seed=seed)
    return FailureProblem(t), t


def test_pairwise_interactions_recovers_ground_truth():
    p, t = _problem(seed=1, n=8)
    inter = pairwise_interactions(p, top=len(t._pairs))
    found = {tuple(sorted((int(a[1:]), int(b[1:])))) for a, b, _ in inter}
    truth = {tuple(sorted(pair)) for pair in t._pairs}
    assert found == truth
    # interaction strength equals the pair weight (1.0 by default)
    for _, _, v in inter:
        assert v > 0.5


def test_marginal_importance_flags_involved_factors():
    p, t = _problem(seed=2, n=8)
    worst = t.worst_config()
    bits = tuple(worst[n] for n in p.search_space.names)
    marg = marginal_importance(p, bits)
    involved = {i for pair in t._pairs for i in pair}
    for i in involved:
        # turning off an involved factor should reduce failure (positive importance)
        assert marg[f"f{i}"] > 0


def test_minimal_critical_set_reduces_or_preserves():
    p, t = _problem(seed=3, n=8)
    worst = t.worst_config()
    bits = tuple(worst[n] for n in p.search_space.names)
    mcs = minimal_critical_set(p, bits, retain=0.5)
    assert mcs["score"] >= 0.5 * mcs["base_score"]
    assert len(mcs["factors"]) <= sum(bits)


def test_marginal_reports_nan_for_infeasible_neighbour():
    t = get_target("sum_threshold", num_factors=5)
    c = ConstraintSet.build(t.make_search_space(), max_active=3)
    p = FailureProblem(t, constraints=c)
    bits = (1, 1, 1, 0, 0)  # exactly at the cardinality boundary
    marg = marginal_importance(p, bits)
    # flipping an inactive factor on would make 4 active -> infeasible -> nan
    nan_count = sum(1 for v in marg.values() if math.isnan(v))
    assert nan_count >= 1


def test_stress_report_markdown_and_json():
    p, t = _problem(seed=4, n=6)
    from quantum_chaos.optimizers import RandomSearch

    r = RandomSearch(seed=0, max_evaluations=1000).optimize(p)
    report = StressReport.build(p, r)
    md = report.to_markdown()
    assert "Stress-test report" in md
    assert "minimal critical fault set" in md
    d = report.to_dict()
    assert set(d.keys()) >= {"target", "result", "critical_set",
                             "marginal_importance", "top_interactions"}


def test_stress_report_save(tmp_path):
    p, t = _problem(seed=5, n=6)
    from quantum_chaos.optimizers import RandomSearch

    r = RandomSearch(seed=0, max_evaluations=800).optimize(p)
    report = StressReport.build(p, r)
    md_path = tmp_path / "r.md"
    json_path = tmp_path / "r.json"
    report.save(str(md_path))
    report.save(str(json_path))
    assert md_path.read_text().startswith("# Stress-test report")
    import json

    assert "critical_set" in json.loads(json_path.read_text())
