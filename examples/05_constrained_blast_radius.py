"""Example 5 — industrial stress test with a blast-radius constraint + report.

The realistic operator question is not "what is the single worst thing that can
happen?" but "what is the worst thing that can happen *if at most K components
fail at once*, given that some fault combinations are impossible?"  This example:

1. builds a cascading-outage target over a service dependency graph,
2. constrains the search to a blast radius (``max_active``) and forbids an
   impossible pairing and requires an implication,
3. runs the Variational Quantum Sampler inside that feasible region, and
4. emits a root-cause stress report (minimal critical fault set + interactions).
"""

from __future__ import annotations

import os

from quantum_chaos import ConstraintSet, FailureProblem, StressReport, get_target
from quantum_chaos.optimizers import VariationalQuantumSampler


def main() -> None:
    target = get_target("cascading_failure", num_factors=12, seed=17)
    space = target.make_search_space()

    # Operational reality: at most 4 simultaneous injected faults; two services
    # can never be down together; and injecting f0 implies f1 also degrades.
    constraints = ConstraintSet.build(
        space,
        max_active=4,
        forbidden_pairs=[("f2", "f3")],
        implications=[("f0", "f1")],
    )
    print(f"feasible fraction of the space: {constraints.feasible_fraction():.3%}")

    problem = FailureProblem(target, constraints=constraints)
    vqs = VariationalQuantumSampler(
        reps=2, shots=256, iterations=60, seed=0, max_evaluations=6000
    )
    result = vqs.optimize(problem)

    print(result.summary())
    print(f"\nfeasible? {constraints.is_feasible(result.best_bits)}  "
          f"(infeasible configs skipped: {problem.num_infeasible})")

    report = StressReport.build(problem, result, retain=0.9)
    out = os.path.join(os.path.dirname(__file__), "05_stress_report.md")
    report.save(out)
    print(f"\nworst-case cascade hits {int(result.best_score)} of 12 services")
    print("critical faults:", report.critical_set["factors"])
    print(f"saved root-cause report -> {out}")


if __name__ == "__main__":
    main()
