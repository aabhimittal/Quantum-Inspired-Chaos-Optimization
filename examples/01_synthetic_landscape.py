"""Example 1 — search a synthetic failure landscape with the VQS optimizer.

Runs the flagship Variational Quantum Sampler against the ``correlated_faults``
landscape, where individual faults are cheap but specific *pairs* interact
catastrophically.  Prints the worst combination found and (if matplotlib is
installed) saves a convergence plot next to this file.
"""

from __future__ import annotations

import os

from quantum_chaos import FailureProblem, get_target
from quantum_chaos.optimizers import VariationalQuantumSampler


def main() -> None:
    target = get_target("correlated_faults", num_factors=8)
    problem = FailureProblem(target)

    vqs = VariationalQuantumSampler(
        reps=2, shots=256, iterations=50, cvar_alpha=0.15,
        backend_name="statevector", seed=0, max_evaluations=4000,
    )
    result = vqs.optimize(problem)

    print(result.summary())
    print("\nInteracting pairs (ground truth):", target._pairs)
    print("Amplitude concentration (CVaR): "
          f"{result.metadata['initial_cvar']:.3f} -> {result.metadata['final_cvar']:.3f}")

    try:
        from quantum_chaos.viz import convergence_plot

        out = os.path.join(os.path.dirname(__file__), "01_convergence.png")
        convergence_plot({"vqs": result}, save_path=out)
        print(f"\nsaved {out}")
    except ImportError:
        print("\n(install the [viz] extra to render plots)")


if __name__ == "__main__":
    main()
