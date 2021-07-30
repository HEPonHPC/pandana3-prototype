import pytest
import pandana3.core.index as index
from pandana3.core.grouping import Grouping


def test_simple_index_creation():
    a = index.SimpleIndex()
    assert a.is_trivial
    assert a.grouping == Grouping()

    b = index.SimpleIndex(False)
    assert not b.is_trivial
    assert b.grouping == Grouping()

    with pytest.raises(TypeError):
        c = index.SimpleIndex(grouping=Grouping(["a", "b"]))


def test_multi_index_creation():
    a = index.MultiIndex()
    assert a.is_trivial
    assert a.grouping == Grouping()

    b = index.MultiIndex(False)
    assert not b.is_trivial
    assert b.grouping == Grouping()

    c = index.MultiIndex(grouping=Grouping(["a", "b"]))
    assert c.is_trivial
    assert c.grouping.column_names == ["a", "b"]
