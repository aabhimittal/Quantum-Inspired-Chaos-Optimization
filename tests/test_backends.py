import numpy as np
import pytest

from quantum_chaos.quantum.backends import (
    AerBackend,
    StatevectorBackend,
    get_backend,
    sample_configs,
)
from quantum_chaos.quantum.circuits import bind, variational_ansatz


def _bound_circuit(seed=0, n=4):
    qc, params = variational_ansatz(n, reps=2)
    vals = np.random.default_rng(seed).uniform(0, np.pi, len(params))
    return bind(qc, params, vals), n


def test_statevector_probabilities_normalized():
    circ, n = _bound_circuit()
    bk = StatevectorBackend(seed=1)
    probs = bk.probabilities(circ)
    assert len(probs) == 2 ** n
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_statevector_sample_counts_sum_to_shots():
    circ, n = _bound_circuit()
    bk = StatevectorBackend(seed=1)
    counts = bk.sample(circ, shots=500)
    assert sum(counts.values()) == 500
    for cfg in counts:
        assert len(cfg) == n


def test_sample_configs_returns_probabilities():
    circ, _ = _bound_circuit()
    bk = StatevectorBackend(seed=2)
    samples = sample_configs(bk, circ, shots=200)
    total = sum(p for _, p in samples)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_aer_backend_matches_distribution_shape():
    circ, n = _bound_circuit()
    aer = AerBackend(seed=3)
    counts = aer.sample(circ, shots=400)
    assert sum(counts.values()) == 400
    # Empirical distribution should broadly track exact probabilities.
    sv = StatevectorBackend(seed=3)
    probs = sv.probabilities(circ)
    top_exact = int(np.argmax(probs))
    from quantum_chaos.core.search_space import config_to_index
    empirical = {config_to_index(c): v for c, v in counts.items()}
    # the exact-mode top state should have appreciable empirical mass
    assert empirical.get(top_exact, 0) > 0


def test_get_backend_factory():
    assert isinstance(get_backend("statevector"), StatevectorBackend)
    assert isinstance(get_backend("aer"), AerBackend)
    with pytest.raises(KeyError):
        get_backend("nope")
