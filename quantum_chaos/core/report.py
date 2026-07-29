"""Assemble a human- and machine-readable stress-test report.

A :class:`StressReport` turns an :class:`~quantum_chaos.core.result.OptimizationResult`
plus root-cause analysis into an audit artifact you can drop into a CI log, a
post-mortem, or a dashboard — the deliverable an engineer actually wants after a
stress run, not just a bitstring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .analysis import marginal_importance, minimal_critical_set, pairwise_interactions
from .result import OptimizationResult


@dataclass
class StressReport:
    """A root-cause audit of a discovered failure mode."""

    target: str
    result: OptimizationResult
    critical_set: Dict[str, object] = field(default_factory=dict)
    marginals: Dict[str, float] = field(default_factory=dict)
    interactions: List[Tuple[str, str, float]] = field(default_factory=list)
    constraint_summary: Dict[str, object] = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        problem,
        result: OptimizationResult,
        retain: float = 0.9,
        top_interactions: int = 5,
    ) -> "StressReport":
        """Run the analyses around ``result.best_bits`` and package them."""
        bits = result.best_bits
        critical = minimal_critical_set(problem, bits, retain=retain)
        marginals = marginal_importance(problem, bits)
        interactions = pairwise_interactions(problem, top=top_interactions)

        constraint_summary: Dict[str, object] = {}
        if problem.constraints is not None:
            c = problem.constraints
            constraint_summary = {
                "max_active": c.max_active,
                "min_active": c.min_active,
                "budget": c.budget,
                "num_forbidden_pairs": len(c.forbidden_pairs),
                "num_mutual_exclusions": len(c.mutual_exclusions),
                "num_implications": len(c.implications),
                "best_is_feasible": c.is_feasible(bits),
                "num_infeasible_seen": problem.num_infeasible,
            }

        return cls(
            target=getattr(problem, "name", "target"),
            result=result,
            critical_set=critical,
            marginals=marginals,
            interactions=interactions,
            constraint_summary=constraint_summary,
        )

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "target": self.target,
            "result": self.result.to_dict(),
            "critical_set": {
                "factors": self.critical_set.get("factors", []),
                "score": self.critical_set.get("score"),
                "base_score": self.critical_set.get("base_score"),
                "retain": self.critical_set.get("retain"),
                "config": self.critical_set.get("config", {}),
            },
            "marginal_importance": self.marginals,
            "top_interactions": [
                {"factors": [a, b], "interaction": v} for a, b, v in self.interactions
            ],
            "constraints": self.constraint_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        r = self.result
        lines: List[str] = []
        lines.append(f"# Stress-test report: `{self.target}`")
        lines.append("")
        lines.append("## Worst-case failure combination")
        lines.append("")
        lines.append(f"- **Optimizer:** `{r.optimizer}`  ·  **Backend:** `{r.backend or 'n/a'}`")
        lines.append(f"- **Failure score:** `{r.best_score:.6g}`")
        lines.append(f"- **Combination:** `{r.best_string}`")
        active = ", ".join(r.active_factors) or "(none)"
        lines.append(f"- **Active faults:** {active}")
        lines.append(
            f"- **Cost:** {r.num_evaluations} distinct evaluations "
            f"/ {r.num_queries} queries"
        )
        if r.metadata.get("best_variance") is not None and r.metadata.get("repeats"):
            lines.append(
                f"- **Noise:** {r.metadata['repeats']} repeats, "
                f"aggregator `{r.metadata.get('aggregator')}`, "
                f"variance `{r.metadata['best_variance']:.4g}`"
            )
        lines.append("")

        # Root cause.
        cs = self.critical_set
        lines.append("## Root cause — minimal critical fault set")
        lines.append("")
        crit = ", ".join(cs.get("factors", [])) or "(none)"
        lines.append(
            f"The smallest fault subset that still reproduces "
            f"≥ {int(100 * cs.get('retain', 0.9))}% of the failure:"
        )
        lines.append("")
        lines.append(f"- **Critical faults:** {crit}")
        lines.append(
            f"- **Reduced score:** `{cs.get('score'):.6g}` "
            f"(of `{cs.get('base_score'):.6g}` full)"
        )
        lines.append("")

        # Marginals.
        lines.append("## Per-factor importance")
        lines.append("")
        lines.append("| factor | importance |")
        lines.append("|---|---|")
        for name, val in sorted(
            self.marginals.items(), key=lambda kv: (kv[1] != kv[1], -(kv[1] if kv[1] == kv[1] else 0))
        ):
            shown = "n/a (infeasible)" if val != val else f"{val:+.4g}"
            lines.append(f"| `{name}` | {shown} |")
        lines.append("")

        # Interactions.
        if self.interactions:
            lines.append("## Strongest fault interactions")
            lines.append("")
            lines.append("| factor A | factor B | interaction |")
            lines.append("|---|---|---|")
            for a, b, v in self.interactions:
                lines.append(f"| `{a}` | `{b}` | {v:+.4g} |")
            lines.append("")

        # Constraints.
        if self.constraint_summary:
            lines.append("## Constraints")
            lines.append("")
            for k, v in self.constraint_summary.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Write Markdown (``.md``) or JSON (any other extension) to ``path``."""
        text = self.to_markdown() if path.endswith(".md") else self.to_json()
        with open(path, "w") as fh:
            fh.write(text)
