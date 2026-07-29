# Quantum-Inspired Chaos Optimization

> Use quantum heuristics to find worst-case failure combinations and stress-test AI systems.

Classical chaos engineering is like throwing rocks at a building to see what
breaks — inefficient, and likely to miss the critical structural weaknesses.
**Quantum-inspired chaos optimization** is like a probability wave that
simultaneously exists everywhere in the building, naturally concentrating where
the structure is weakest, then collapsing to reveal the exact spots most likely
to cause catastrophic failure.

This framework puts a quantum *superposition* over the space of **failure
configurations** — which faults are active — lets amplitude concentrate on a
system's weak spots, and measures (collapses) to surface the exact combinations
most likely to break it. It runs on real Qiskit (Aer simulator) and works
against *any* black-box target through a simple `callable(config) -> failure_score`
interface.

---

## Why combinations?

The hardest failures in AI systems are rarely caused by a single bad input. They
emerge when several individually-harmless perturbations **interact**: a tokenizer
quirk *plus* a unicode trick *plus* an emoji injection. The number of such
combinations grows as `2^N`, far too many to test exhaustively. Quantum search
is built for exactly this: exploring an exponential configuration space and
concentrating probability on the worst corners.

---

## Install

```bash
# core + real quantum SDK (Qiskit + Aer)
pip install -e ".[quantum]"

# everything: quantum, plots, tests
pip install -e ".[all]"
```

Python ≥ 3.9. The default `statevector` backend is exact and fast; the `aer`
backend is the shot-based "real SDK" path closest to hardware.

---

## Quickstart

Point it at any function that scores how badly your system fails:

```python
from quantum_chaos import FailureProblem, FunctionTarget
from quantum_chaos.optimizers import VariationalQuantumSampler

FACTORS = ["homoglyphs", "zero_width", "leetspeak", "emoji", "base64", "prefix"]

def my_ai_system_failure(config: dict) -> float:
    on = {k for k, v in config.items() if v}
    failure = 0.05 * len(on)
    if {"homoglyphs", "zero_width"} <= on:   # an interaction the model can't handle
        failure += 0.5
    return failure

target  = FunctionTarget(my_ai_system_failure, num_factors=6, factor_names=FACTORS)
problem = FailureProblem(target)
result  = VariationalQuantumSampler(seed=0).optimize(problem)

print(result.summary())
print("worst-case combination:", result.active_factors)
```

---

## Command line

```bash
qchaos list-targets        # built-in synthetic failure landscapes
qchaos list-optimizers

# search one target with one optimizer, print JSON
qchaos run --target correlated_faults --factors 10 --optimizer vqs --seed 0

# race several optimizers on the same landscape
qchaos benchmark --target deceptive_trap --factors 12 \
    --optimizers vqs,qaoa,annealing,random,hillclimb --plot bench.png
```

---

## The optimizers

| Optimizer     | Idea                                                                                   | Best for |
|---------------|----------------------------------------------------------------------------------------|----------|
| **`vqs`**     | *Variational Quantum Sampler* — a parameterized circuit is a probability wave over failure configs; a CVaR objective + SPSA reshape it to pile amplitude on the worst tail. | Any black box; the flagship. |
| **`qaoa`**    | Fits a **QUBO surrogate** of the landscape, trains QAOA angles to its worst corner, verifies on the true target. | Structured / near-quadratic landscapes. |
| **`grover`**  | **Amplitude amplification**: given a threshold recognizer, finds a catastrophic config in `~√(N/M)` oracle queries vs `~N/M` classical. | A hidden "needle" you can recognize but not locate. |
| **`annealing`** | Quantum-inspired simulated annealing with a **transverse-field** schedule enabling multi-bit tunneling out of deceptive traps. | Rugged landscapes; strong baseline. |
| **`random` / `hillclimb`** | Classical baselines for **honest** comparison.                            | Sanity checks. |

The **CVaR** (Conditional Value at Risk) objective is what makes `vqs` hunt the
*worst case* rather than the average: it optimizes the mean of the worst tail of
sampled failures, so amplitude flows toward catastrophes, not typical behavior.

---

## Built-in failure landscapes

Each ships with a known ground-truth worst config, so tests and benchmarks can
verify an optimizer actually found the darkest corner:

- **`correlated_faults`** — single faults are cheap; specific *pairs* interact catastrophically. The central thesis.
- **`deceptive_trap`** — a strong local attractor lures naive search away from the true catastrophe.
- **`needle`** — one hidden catastrophic combination; everything else flat. Showcase for Grover.
- **`sum_threshold`** — smooth monotone degradation; a graceful sanity baseline.
- **`maxsat`** — a planted weighted MAX-2-SAT instance: failure = weight of violated robustness constraints.
- **`cascading_failure`** — an outage that *propagates* across a service dependency graph; failure = blast radius. Pairs naturally with cardinality constraints.

---

## Industrial features

Real stress tests are messier than a clean optimization problem. The framework
handles that head-on.

### Feasibility constraints & blast radius

Not every fault combination is reachable, and operators usually care about the
worst case *within a blast radius* — at most `K` simultaneous faults. A
`ConstraintSet` expresses cardinality/budget limits, mutual exclusions, forbidden
pairs, implications, per-factor weights, and custom predicates. Infeasible
configurations are penalized and **never executed on the target**, and sampling
is confined to the feasible region so even a 17%-feasible space searches well.

```python
from quantum_chaos import ConstraintSet, FailureProblem, get_target
from quantum_chaos.optimizers import VariationalQuantumSampler

target = get_target("cascading_failure", num_factors=12)
constraints = ConstraintSet.build(
    target.make_search_space(),
    max_active=4,                       # blast radius: ≤ 4 faults at once
    forbidden_pairs=[("f2", "f3")],     # impossible together
    implications=[("f0", "f1")],        # f0 failing degrades f1 too
)
problem = FailureProblem(target, constraints=constraints)
result  = VariationalQuantumSampler(seed=0).optimize(problem)
# → a 4-fault injection that cascades across the whole dependency graph
```

```bash
qchaos run --target cascading_failure --factors 12 --max-active 4 --report audit.md
```

### Robust evaluation of noisy targets

AI systems rarely fail deterministically. Set `repeats` and pick a tail-aware
`aggregator` (`mean`, `p95`, `cvar`, …); per-config variance is tracked. Pathological
targets are handled by explicit policy — including the industrially useful
`on_error="as_failure"`, which treats a target that **crashes** on an input as a
catastrophic failure to be discovered.

```python
FailureProblem(target, repeats=16, aggregator="cvar",      # worst-tail risk
               on_error="as_failure", on_nonfinite="clip")
```

### Root-cause analysis & stress reports

Finding a worst case is half the job; you also need to know *which* faults matter.

- `minimal_critical_set` — the smallest fault subset that still reproduces the failure.
- `marginal_importance` — each factor's contribution.
- `pairwise_interactions` — the synergistic fault pairs (recovers ground-truth interactions exactly on the synthetic landscapes).
- `StressReport` — packages all of the above into Markdown or JSON for a CI log or post-mortem.

```python
from quantum_chaos import StressReport
StressReport.build(problem, result).save("audit.md")
```

---

## How it works

```
   failure factors                parameterized circuit             measurement
   (which faults?)   ──encode──►   |ψ(θ)⟩ superposition   ──train──► collapse to
   f0 f1 … f_{N-1}                 over all 2^N configs    (SPSA)     worst combos
        │                               │                                │
        └────────────── target failure_score(config) ◄──────────────────┘
                         (higher = worse; guides θ)
```

1. **Encode** — each binary chaos factor maps to one qubit; a failure
   configuration is an `N`-bit string, one corner of a `2^N` space.
2. **Superpose** — a hardware-efficient ansatz (`vqs`) or QAOA circuit prepares a
   probability wave over every configuration at once.
3. **Concentrate** — measure, score the samples on the real target, and let SPSA
   nudge the circuit so amplitude piles onto the highest-failure corners.
4. **Collapse** — measuring the trained circuit yields the worst-case failure
   combinations.

See `examples/` for runnable walk-throughs:

- `01_synthetic_landscape.py` — VQS on an interaction landscape, with a convergence plot.
- `02_qaoa_vs_random.py` — QAOA vs. random search, side by side.
- `03_blackbox_target.py` — wrap an arbitrary Python function (a mock AI system).
- `04_grover_darkest_corner.py` — Grover finds a hidden catastrophe with a quadratic speedup.
- `05_constrained_blast_radius.py` — constrained cascading-outage search + a root-cause stress report.

---

## Architecture

```
quantum_chaos/
  core/        SearchSpace · FailureProblem (cached, budgeted, noisy, constrained)
               OptimizationResult · ConstraintSet · analysis · StressReport
  targets/     TargetSystem · FunctionTarget · synthetic landscapes · registry
  quantum/     Qiskit backends (statevector / Aer) · ansätze · QUBO surrogate + Ising
  optimizers/  vqs · qaoa · grover · annealing · random · hillclimb
  viz/         convergence · interaction heatmap · amplitude concentration · benchmark bars
  cli.py       qchaos {list-targets,list-optimizers,run,benchmark}
```

**Budget & fairness.** `FailureProblem` memoizes evaluations. It reports two
numbers: `num_evaluations` (distinct configs actually executed — the efficiency
metric) and `num_queries` (every request, the budget currency). Optimizers are
bounded by queries, so local searches always terminate even on tiny spaces.

**Real quantum hardware.** Everything runs on the Aer simulator; swapping in an
IBM Quantum runtime sampler inside `quantum/backends.py` is all it takes to target
a real device (subject to qubit counts).

---

## Development

```bash
pip install -e ".[all]"
pytest -q
```

---

## License

MIT © Abhishek Mittal
