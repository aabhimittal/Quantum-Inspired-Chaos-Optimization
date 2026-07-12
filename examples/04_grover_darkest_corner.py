"""Example 4 — Grover amplitude amplification finds the hidden catastrophe.

The ``needle`` landscape is flat everywhere except one hidden configuration that
is catastrophic — the worst possible case for random search.  Given a recognizer
that can tell "this configuration is catastrophic" (a threshold oracle), Grover's
algorithm amplifies the needle's amplitude and measures it in ``~sqrt(N/M)``
oracle queries instead of the ``~N/M`` a classical scan needs.

The printout reports the theoretical query counts side by side so the quadratic
speedup is explicit.
"""

from __future__ import annotations

from quantum_chaos import FailureProblem, get_target
from quantum_chaos.optimizers import GroverSearch


def main() -> None:
    target = get_target("needle", num_factors=12)
    problem = FailureProblem(target)

    grover = GroverSearch(shots=1024, seed=0)
    result = grover.optimize(problem)

    print(result.summary())
    meta = result.metadata
    print("\nAmplitude-amplification accounting:")
    print(f"  space size N         : {meta['space_size']}")
    print(f"  marked catastrophes M: {meta['num_marked']}")
    print(f"  Grover oracle queries: {meta['grover_oracle_queries']}")
    print(f"  classical expected   : {meta['classical_expected_queries']:.0f}")
    print(f"  speedup factor       : {meta['speedup_factor']:.1f}x")

    worst = target.worst_config()
    found = tuple(worst[n] for n in result.best_config) == result.best_bits
    print(f"\nFound the planted needle: {found}")


if __name__ == "__main__":
    main()
