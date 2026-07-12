"""Matplotlib visualizations of the failure landscape and the search.

All functions degrade gracefully: if matplotlib is missing they raise a clear
``ImportError`` pointing at the ``[viz]`` extra.  Each returns the Matplotlib
``Figure`` and, when given ``save_path``, also writes a PNG.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.result import OptimizationResult
from ..core.search_space import index_to_config


def _require_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install with:\n"
            "    pip install 'quantum-chaos[viz]'"
        ) from exc


def convergence_plot(
    results: Dict[str, OptimizationResult],
    save_path: Optional[str] = None,
    title: str = "Worst-case failure discovered vs. evaluations",
):
    """Overlay best-so-far failure curves for one or more optimizers."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        curve = res.convergence()
        ax.plot(range(1, len(curve) + 1), curve, label=name, linewidth=2)
    ax.set_xlabel("distinct target evaluations")
    ax.set_ylabel("best failure score found")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def landscape_heatmap(
    problem,
    factor_x: int = 0,
    factor_y: int = 1,
    baseline: Optional[Sequence[int]] = None,
    save_path: Optional[str] = None,
):
    """Heatmap of failure over two factors, others held at ``baseline``.

    Useful to *see* an interaction: a bright corner where two factors are both
    active but neither is dangerous alone.
    """
    plt = _require_mpl()
    n = problem.num_factors
    base = list(baseline) if baseline is not None else [0] * n
    grid = np.zeros((2, 2))
    for xv in (0, 1):
        for yv in (0, 1):
            bits = list(base)
            bits[factor_x] = xv
            bits[factor_y] = yv
            grid[yv, xv] = problem.evaluate(tuple(bits))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(grid, origin="lower", cmap="magma", aspect="auto")
    names = problem.search_space.names
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel(names[factor_x])
    ax.set_ylabel(names[factor_y])
    ax.set_title("Failure interaction (2-factor slice)")
    for xv in (0, 1):
        for yv in (0, 1):
            ax.text(xv, yv, f"{grid[yv, xv]:.2f}", ha="center", va="center",
                    color="white")
    fig.colorbar(im, ax=ax, label="failure score")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def benchmark_bar(
    results: Dict[str, OptimizationResult],
    save_path: Optional[str] = None,
):
    """Bar chart comparing the worst failure each optimizer found."""
    plt = _require_mpl()
    names = list(results.keys())
    scores = [results[n].best_score for n in names]
    evals = [results[n].num_evaluations for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, scores, color="#5B8FF9")
    ax.set_ylabel("worst failure score found")
    ax.set_title("Optimizer comparison")
    for bar, ev in zip(bars, evals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{ev} evals", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def amplitude_concentration(
    probabilities: np.ndarray,
    scores: np.ndarray,
    top: int = 20,
    save_path: Optional[str] = None,
):
    """Show measurement probability piling onto the highest-failure configs.

    ``probabilities`` and ``scores`` are both indexed by basis-state integer
    (factor 0 as the LSB).  Bars are the ``top`` configurations by failure
    score; height is the probability the trained circuit assigns them.
    """
    plt = _require_mpl()
    order = np.argsort(scores)[::-1][:top]
    n = int(np.log2(len(probabilities)))
    labels = ["".join(str(b) for b in index_to_config(int(i), n)) for i in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(order)), probabilities[order], color="#E8684A")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("measurement probability")
    ax.set_xlabel("configurations (worst failure first)")
    ax.set_title("Amplitude concentrated on the darkest corners")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig
