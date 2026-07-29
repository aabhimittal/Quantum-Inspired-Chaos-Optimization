"""``qchaos`` command-line interface.

Subcommands
-----------
* ``qchaos list-targets`` / ``qchaos list-optimizers`` — discover what's built in.
* ``qchaos run`` — search one target with one optimizer, print/emit JSON.
* ``qchaos benchmark`` — race several optimizers on the same target.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Dict, List, Optional

from .core.problem import FailureProblem
from .optimizers import OPTIMIZERS, get_optimizer, list_optimizers
from .targets import get_target, list_targets, target_descriptions


def _build_optimizer(name: str, args) -> object:
    """Instantiate an optimizer passing only the kwargs it accepts."""
    cls = OPTIMIZERS[name]
    sig = inspect.signature(cls.__init__)
    candidates = {
        "seed": args.seed,
        "max_evaluations": args.budget,
        "backend_name": args.backend,
        "reps": args.reps,
        "shots": args.shots,
        "iterations": args.iterations,
    }
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters and v is not None}
    return cls(**kwargs)


def _target_kwargs(args) -> Dict:
    kwargs: Dict = {}
    if args.factors is not None:
        kwargs["num_factors"] = args.factors
    if getattr(args, "seed", None) is not None:
        # Only pass seed to targets that accept it.
        pass
    return kwargs


def _make_problem(args) -> FailureProblem:
    tkwargs = _target_kwargs(args)
    # Filter to kwargs the target factory accepts.
    from .targets.registry import _REGISTRY

    factory = _REGISTRY[args.target]
    sig = inspect.signature(factory)
    tkwargs = {k: v for k, v in tkwargs.items() if k in sig.parameters}
    target = get_target(args.target, **tkwargs)

    # Optional feasibility constraints (blast-radius cardinality budget).
    constraints = None
    max_active = getattr(args, "max_active", None)
    if max_active is not None:
        from .core.constraints import ConstraintSet

        constraints = ConstraintSet.build(target.make_search_space(), max_active=max_active)

    problem = FailureProblem(
        target,
        constraints=constraints,
        repeats=getattr(args, "repeats", None) or 1,
        aggregator=getattr(args, "aggregator", None) or "mean",
    )
    return problem, target


def _hit(target, result) -> Optional[bool]:
    worst = target.worst_config()
    if worst is None:
        return None
    names = list(result.best_config.keys())
    wc = tuple(worst[n] for n in names)
    return wc == result.best_bits


def cmd_list_targets(args) -> int:
    descs = target_descriptions()
    print("Available targets:")
    for name, desc in descs.items():
        print(f"  {name:20s} {desc}")
    return 0


def cmd_list_optimizers(args) -> int:
    print("Available optimizers:")
    for name in list_optimizers():
        print(f"  {name}")
    return 0


def cmd_run(args) -> int:
    problem, target = _make_problem(args)
    optimizer = _build_optimizer(args.optimizer, args)
    result = optimizer.optimize(problem)

    hit = _hit(target, result)
    payload = result.to_dict()
    if hit is not None:
        payload["found_known_worst"] = hit

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")
    else:
        print(json.dumps(payload, indent=2))

    if args.plot:
        from .viz.plots import convergence_plot

        convergence_plot({args.optimizer: result}, save_path=args.plot)
        print(f"wrote {args.plot}")

    if getattr(args, "report", None):
        from .core.report import StressReport

        report = StressReport.build(problem, result)
        report.save(args.report)
        print(f"wrote {args.report}")
    return 0


def cmd_benchmark(args) -> int:
    names = args.optimizers.split(",") if args.optimizers else list_optimizers()
    results = {}
    target = None
    for opt_name in names:
        opt_name = opt_name.strip()
        if opt_name not in OPTIMIZERS:
            print(f"skipping unknown optimizer {opt_name!r}", file=sys.stderr)
            continue
        problem, target = _make_problem(args)
        try:
            optimizer = _build_optimizer(opt_name, args)
            results[opt_name] = optimizer.optimize(problem)
        except Exception as exc:  # keep the race going if one optimizer errors
            print(f"{opt_name} failed: {exc}", file=sys.stderr)

    if not results:
        print("no optimizers ran", file=sys.stderr)
        return 1

    header = f"{'optimizer':12s} {'best_score':>12s} {'distinct':>9s} {'queries':>9s} {'worst?':>7s}"
    print(header)
    print("-" * len(header))
    for name, res in sorted(results.items(), key=lambda kv: -kv[1].best_score):
        hit = _hit(target, res)
        mark = "" if hit is None else ("yes" if hit else "no")
        print(
            f"{name:12s} {res.best_score:12.4f} {res.num_evaluations:9d} "
            f"{res.num_queries:9d} {mark:>7s}"
        )

    if args.plot:
        from .viz.plots import convergence_plot

        convergence_plot(results, save_path=args.plot,
                         title=f"Failure discovery on '{args.target}'")
        print(f"wrote {args.plot}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qchaos",
        description="Quantum-inspired chaos optimization: find worst-case AI failure modes.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list-targets", help="list built-in failure landscapes").set_defaults(
        func=cmd_list_targets
    )
    sub.add_parser("list-optimizers", help="list available optimizers").set_defaults(
        func=cmd_list_optimizers
    )

    def add_common(sp):
        sp.add_argument("--target", default="correlated_faults", help="target name")
        sp.add_argument("--factors", type=int, default=None, help="number of chaos factors")
        sp.add_argument("--backend", default="statevector",
                        choices=["statevector", "aer"], help="quantum backend")
        sp.add_argument("--budget", type=int, default=None, help="max target queries")
        sp.add_argument("--seed", type=int, default=None, help="random seed")
        sp.add_argument("--reps", type=int, default=None, help="ansatz/QAOA layers")
        sp.add_argument("--shots", type=int, default=None, help="measurement shots")
        sp.add_argument("--iterations", type=int, default=None,
                        help="optimizer iterations")
        sp.add_argument("--max-active", type=int, default=None, dest="max_active",
                        help="blast-radius constraint: at most this many faults active")
        sp.add_argument("--repeats", type=int, default=None,
                        help="evaluate a noisy target this many times per config")
        sp.add_argument("--aggregator", default=None,
                        choices=["mean", "max", "min", "median", "p95", "p05", "cvar"],
                        help="how to reduce noisy repeats")

    run_p = sub.add_parser("run", help="run one optimizer on one target")
    add_common(run_p)
    run_p.add_argument("--optimizer", default="vqs", choices=list(OPTIMIZERS))
    run_p.add_argument("--json", default=None, help="write result JSON to this path")
    run_p.add_argument("--plot", default=None, help="write a convergence PNG here")
    run_p.add_argument("--report", default=None,
                       help="write a root-cause stress report (.md or .json) here")
    run_p.set_defaults(func=cmd_run)

    bench_p = sub.add_parser("benchmark", help="compare optimizers on a target")
    add_common(bench_p)
    bench_p.add_argument("--optimizers", default="vqs,qaoa,annealing,random,hillclimb",
                         help="comma-separated optimizer names")
    bench_p.add_argument("--plot", default=None, help="write a convergence PNG here")
    bench_p.set_defaults(func=cmd_benchmark)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
