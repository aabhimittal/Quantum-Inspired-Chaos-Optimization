"""Example 3 — wrap an arbitrary Python function as a target (the common case).

This is how you point the framework at a *real* AI system.  Define a function
that takes a fault configuration and returns a failure score (higher = worse) —
here a mock "content-moderation model" whose accuracy collapses under specific
combinations of input perturbations — wrap it in ``FunctionTarget``, and let the
Variational Quantum Sampler hunt for the worst-case combination.
"""

from __future__ import annotations

from quantum_chaos import FailureProblem, FunctionTarget
from quantum_chaos.optimizers import VariationalQuantumSampler

# The knobs we are allowed to turn while stress-testing the system.
FACTORS = [
    "unicode_homoglyphs",
    "zero_width_spaces",
    "leetspeak",
    "emoji_injection",
    "base64_wrap",
    "prompt_prefix",
]


def mock_model_failure(config: dict) -> float:
    """Pretend to run an AI system and return how badly it failed (0..1+).

    The system is individually robust to each perturbation, but three nasty
    interactions cause a jailbreak (high failure):

    * homoglyphs + zero-width spaces defeat the tokenizer,
    * leetspeak + base64 wrapping hides the payload,
    * emoji injection amplifies whatever else is active.
    """
    on = {k for k, v in config.items() if v}
    failure = 0.05 * len(on)  # mild baseline degradation
    if {"unicode_homoglyphs", "zero_width_spaces"} <= on:
        failure += 0.5
    if {"leetspeak", "base64_wrap"} <= on:
        failure += 0.4
    if "emoji_injection" in on:
        failure *= 1.3
    return failure


def main() -> None:
    target = FunctionTarget(
        mock_model_failure,
        num_factors=len(FACTORS),
        name="mock_moderation_model",
        factor_names=FACTORS,
        description="mock content-moderation model with interaction failures",
    )
    problem = FailureProblem(target)

    vqs = VariationalQuantumSampler(reps=2, shots=256, iterations=60, seed=7)
    result = vqs.optimize(problem)

    print(result.summary())
    print("\nWorst-case perturbation combination the search discovered:")
    for name in result.active_factors:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
