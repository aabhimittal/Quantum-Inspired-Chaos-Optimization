"""The object every optimizer returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Config = Tuple[int, ...]


@dataclass
class OptimizationResult:
    """The outcome of a failure-mode search.

    Attributes
    ----------
    best_bits:
        The worst-case failure combination found (canonical bit tuple).
    best_score:
        Its failure severity (higher = worse).
    best_config:
        ``{factor_name: 0/1}`` view of ``best_bits``.
    history:
        Chronological ``(bits, score)`` of every *distinct* configuration the
        optimizer evaluated, in the order they were first seen.
    num_evaluations:
        Number of real (cache-miss) target evaluations spent.
    optimizer / backend:
        Human-readable provenance labels.
    metadata:
        Free-form extra info (iterations, hyper-parameters, comparison stats).
    """

    best_bits: Config
    best_score: float
    best_config: Dict[str, int]
    history: List[Tuple[Config, float]] = field(default_factory=list)
    num_evaluations: int = 0
    num_queries: int = 0
    optimizer: str = ""
    backend: str = ""
    metadata: Dict = field(default_factory=dict)

    # -- derived views ----------------------------------------------------
    @property
    def best_string(self) -> str:
        return "".join(str(b) for b in self.best_bits)

    @property
    def active_factors(self) -> List[str]:
        return [name for name, bit in self.best_config.items() if bit]

    def convergence(self) -> List[float]:
        """Best-so-far failure score after each distinct evaluation."""
        best = float("-inf")
        curve: List[float] = []
        for _, score in self.history:
            if score > best:
                best = score
            curve.append(best)
        return curve

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer,
            "backend": self.backend,
            "best_string": self.best_string,
            "best_bits": list(self.best_bits),
            "best_score": self.best_score,
            "best_config": self.best_config,
            "active_factors": self.active_factors,
            "num_evaluations": self.num_evaluations,
            "num_queries": self.num_queries,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        active = ", ".join(self.active_factors) or "(none)"
        lines = [
            f"optimizer      : {self.optimizer}",
            f"backend        : {self.backend}",
            f"worst combo    : {self.best_string}",
            f"failure score  : {self.best_score:.6g}",
            f"active factors : {active}",
            f"evaluations    : {self.num_evaluations} distinct / {self.num_queries} queries",
        ]
        return "\n".join(lines)
