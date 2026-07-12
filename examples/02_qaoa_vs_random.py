"""Example 2 — QAOA vs. random search on the same failure landscape.

QAOA fits a QUBO surrogate of the failure landscape, trains its angles to
concentrate probability on the worst corner, then verifies on the true target.
We race it against plain random search under the same query budget and print a
side-by-side comparison — an honest look at whether the quantum-inspired method
finds a *worse* failure with *fewer* real evaluations.
"""

from __future__ import annotations

import os

from quantum_chaos import FailureProblem, get_target
from quantum_chaos.optimizers import QAOAOptimizer, RandomSearch


def main() -> None:
    results = {}
    for name, opt in {
        "qaoa": QAOAOptimizer(reps=2, iterations=80, surrogate_samples=150, seed=1),
        "random": RandomSearch(max_evaluations=800, seed=1),
    }.items():
        problem = FailureProblem(get_target("correlated_faults", num_factors=10))
        results[name] = opt.optimize(problem)

    for name, res in results.items():
        print(f"{name:8s} worst={res.best_score:7.3f} "
              f"distinct={res.num_evaluations:4d} queries={res.num_queries:4d} "
              f"combo={res.best_string}")

    try:
        from quantum_chaos.viz import benchmark_bar

        out = os.path.join(os.path.dirname(__file__), "02_qaoa_vs_random.png")
        benchmark_bar(results, save_path=out)
        print(f"\nsaved {out}")
    except ImportError:
        print("\n(install the [viz] extra to render plots)")


if __name__ == "__main__":
    main()
