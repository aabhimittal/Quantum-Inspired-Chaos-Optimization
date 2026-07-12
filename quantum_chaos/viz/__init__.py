"""Visualization helpers (optional; requires the ``[viz]`` extra)."""

from .plots import (
    amplitude_concentration,
    benchmark_bar,
    convergence_plot,
    landscape_heatmap,
)

__all__ = [
    "convergence_plot",
    "landscape_heatmap",
    "benchmark_bar",
    "amplitude_concentration",
]
