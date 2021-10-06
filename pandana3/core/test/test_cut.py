from __future__ import annotations
import pytest
import pandas as pd
import h5py as h5
from pandana3.core.cut import SimpleCut
from pandana3.core.var import SimpleVar
from pandana3.core.index import SimpleIndex


@pytest.fixture()
def c00() -> SimpleCut:
    """"A cut containing a SimpleVar of electrons with pt and phi."""
    return SimpleCut(SimpleVar("electrons", ["pt", "phi"]), lambda df: df["pt"] > 50.0)


def test_constructed_is_not_prepared(c00: SimpleCut) -> None:
    assert not c00.prepared


def test_tables_read(c00: SimpleCut) -> None:
    assert c00.inq_tables_read() == {"electrons"}


def test_prepare(c00: SimpleCut, datafile: h5.File) -> None:
    c00.prepare(datafile)
    assert c00.prepared


def test_eval(c00: SimpleCut, datafile: h5.File) -> None:
    c00.prepare(datafile)
    assert c00.prepared
    assert c00.inq_datasets_read() == {"/electrons/pt", "/electrons/phi"}
    vals = c00.eval(datafile)
    assert isinstance(vals, pd.Series)
    assert len(vals) == 29
    assert sum(vals) == 4


def test_resolve_metadata(c00: SimpleCut, datafile: h5.File) -> None:
    c00.prepare(datafile)
    available, needed = c00.resolve_metadata(datafile)
    assert available == ["evtnum", "electrons_idx"]
    assert needed == []
