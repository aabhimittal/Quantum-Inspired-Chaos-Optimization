import pytest

from quantum_chaos.core.search_space import (
    SearchSpace,
    bitstring_to_config,
    config_to_index,
    index_to_config,
)


def test_named_and_unnamed_spaces():
    a = SearchSpace(3)
    assert a.names == ["f0", "f1", "f2"]
    b = SearchSpace(["x", "y"])
    assert b.num_factors == 2
    assert b.names == ["x", "y"]


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        SearchSpace(["a", "a"])


def test_as_config_from_various_forms():
    s = SearchSpace(["a", "b", "c"])
    assert s.as_config("101") == (1, 0, 1)
    assert s.as_config([1, 0, 1]) == (1, 0, 1)
    assert s.as_config({"a": 1, "c": 1}) == (1, 0, 1)


def test_string_left_is_factor_zero():
    s = SearchSpace(3)
    # bits[0]=1 -> leftmost char is '1'
    assert s.to_string((1, 0, 0)) == "100"


def test_decode_and_active_factors():
    s = SearchSpace(["a", "b", "c"])
    assert s.decode("110") == {"a": 1, "b": 1, "c": 0}
    assert s.active_factors("110") == ["a", "b"]


def test_bitstring_to_config_reverses_qiskit_endianness():
    # Qiskit key "100" has qubit 0 on the right (=0), qubit 2 on the left (=1).
    assert bitstring_to_config("100", 3) == (0, 0, 1)


def test_index_roundtrip():
    for idx in range(8):
        bits = index_to_config(idx, 3)
        assert config_to_index(bits) == idx


def test_all_configs_enumerates_space():
    s = SearchSpace(3)
    assert len(list(s.all_configs())) == 8
