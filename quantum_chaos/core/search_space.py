"""Encoding of the *failure configuration* search space.

A :class:`SearchSpace` describes a set of ``N`` binary **chaos factors** — the
independent fault dimensions we are allowed to toggle when trying to break a
target system (e.g. "drop 10% of packets", "corrupt token 7", "spike latency").
Each factor maps to exactly one qubit, so a full failure configuration is a
length-``N`` bitstring and the whole space has ``2**N`` corners.

Bit convention (important, used everywhere):

* A *config* is a tuple of ``0``/``1`` ints, ``bits[i]`` belonging to factor ``i``.
* When we render a config as a string we use ``bits[0]`` on the **left**:
  ``(1, 0, 1) -> "101"``.  Qiskit measurement keys are little-endian (qubit 0 on
  the right), so any conversion from raw Qiskit counts must reverse the string.
  :func:`bitstring_to_config` performs that reversal in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np

Config = Tuple[int, ...]
ConfigLike = Union[str, Sequence[int], Dict[str, int]]


@dataclass(frozen=True)
class ChaosFactor:
    """A single binary fault dimension."""

    name: str
    description: str = ""


@dataclass
class SearchSpace:
    """A space of ``N`` binary chaos factors, one qubit per factor."""

    factors: List[ChaosFactor] = field(default_factory=list)

    def __init__(self, factors: Union[int, Sequence[Union[str, ChaosFactor]]]):
        if isinstance(factors, int):
            resolved = [ChaosFactor(name=f"f{i}") for i in range(factors)]
        else:
            resolved = []
            for item in factors:
                if isinstance(item, ChaosFactor):
                    resolved.append(item)
                else:
                    resolved.append(ChaosFactor(name=str(item)))
            if not resolved:
                raise ValueError("SearchSpace needs at least one chaos factor")
        # dataclass with a custom __init__: set the field explicitly.
        object.__setattr__(self, "factors", resolved)
        names = [f.name for f in resolved]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate chaos factor names: {names}")

    # -- basic properties -------------------------------------------------
    @property
    def num_factors(self) -> int:
        return len(self.factors)

    #: number of qubits equals number of binary factors
    num_qubits = num_factors

    @property
    def names(self) -> List[str]:
        return [f.name for f in self.factors]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_factors

    # -- encode / decode --------------------------------------------------
    def as_config(self, value: ConfigLike) -> Config:
        """Coerce many representations into the canonical ``bits`` tuple."""
        n = self.num_factors
        if isinstance(value, str):
            if len(value) != n:
                raise ValueError(f"expected {n}-char bitstring, got {value!r}")
            return tuple(int(c) for c in value)
        if isinstance(value, dict):
            return tuple(int(bool(value.get(name, 0))) for name in self.names)
        seq = list(value)
        if len(seq) != n:
            raise ValueError(f"expected {n} bits, got {len(seq)}")
        return tuple(int(bool(b)) for b in seq)

    def decode(self, value: ConfigLike) -> Dict[str, int]:
        """Return a ``{factor_name: 0/1}`` view of a configuration."""
        bits = self.as_config(value)
        return {name: bit for name, bit in zip(self.names, bits)}

    def to_string(self, value: ConfigLike) -> str:
        """Canonical string form, ``bits[0]`` on the left."""
        return "".join(str(b) for b in self.as_config(value))

    def active_factors(self, value: ConfigLike) -> List[str]:
        """Names of the factors that are switched on in ``value``."""
        return [name for name, bit in self.decode(value).items() if bit]

    # -- sampling / enumeration ------------------------------------------
    def random_config(self, rng: np.random.Generator) -> Config:
        return tuple(int(b) for b in rng.integers(0, 2, size=self.num_factors))

    def all_configs(self) -> Iterator[Config]:
        """Enumerate every corner of the space (only sane for small ``N``)."""
        for bits in product((0, 1), repeat=self.num_factors):
            yield bits


def bitstring_to_config(key: str, num_factors: int) -> Config:
    """Convert a raw Qiskit measurement key into a canonical config tuple.

    Qiskit keys are little-endian: the right-most character is qubit ``0``.
    We reverse so that ``config[i]`` corresponds to qubit/factor ``i``.
    """
    cleaned = key.replace(" ", "")
    if len(cleaned) != num_factors:
        raise ValueError(
            f"measurement key {key!r} has {len(cleaned)} bits, expected {num_factors}"
        )
    return tuple(int(c) for c in reversed(cleaned))


def config_to_index(bits: Sequence[int]) -> int:
    """Integer index of a config with ``bits[0]`` as the least-significant bit."""
    idx = 0
    for i, b in enumerate(bits):
        if b:
            idx |= 1 << i
    return idx


def index_to_config(index: int, num_factors: int) -> Config:
    """Inverse of :func:`config_to_index`."""
    return tuple((index >> i) & 1 for i in range(num_factors))
