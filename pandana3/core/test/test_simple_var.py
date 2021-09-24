import pytest
from pandana3.core.var import SimpleVar
import h5py as h5
import pandas as pd
import numpy as np


@pytest.fixture()
def sv00() -> SimpleVar:
    return SimpleVar("electrons", ["pt", "eta"])


@pytest.fixture()
def dummyfile() -> h5.File:
    # Note: we open in write mode, so that there doesn't need to be a file
    # on the filesystem, and with backing_store=False, so that we don't
    # create one when the h5.File is closed.
    return h5.File("dummy.h5", mode="w", driver="core", backing_store=False)


@pytest.fixture()
def datafile() -> h5.File:
    f = h5.File("small.h5", "r")
    assert f
    return f


def test_multiple_table_names():
    with pytest.raises(TypeError):
        _ = SimpleVar(["electrons", "muons"], ["pt"])


def test_column_names_not_list():
    with pytest.raises(TypeError):
        _ = SimpleVar("electrons", "pt")


def test_column_names_empy_list():
    with pytest.raises(ValueError):
        _ = SimpleVar("electrons", [])


def test_right_columns(sv00: SimpleVar) -> None:
    assert sv00.columns == ["pt", "eta"]


def test_evaluate_unprepared(sv00: SimpleVar, dummyfile: h5.File) -> None:
    with pytest.raises(ValueError):
        _ = sv00.eval(dummyfile)


def test_newly_constructed_is_unprepared(sv00: SimpleVar) -> None:
    assert not sv00.prepared


def test_newly_constructed_indices(sv00: SimpleVar) -> None:
    assert len(sv00.required_indices) == 0


def test_prepare_sets_state(sv00: SimpleVar, datafile: h5.File) -> None:
    sv00.prepare(datafile)
    assert sv00.prepared


def test_eval_prepared_no_index(sv00: SimpleVar, datafile: h5.File) -> None:
    sv00.prepare(datafile)
    df = sv00.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["pt", "eta"]
    assert isinstance(df.index, pd.RangeIndex)
    assert len(df) == 29


def test_set_required_indices(sv00: SimpleVar) -> None:
    sv00.set_required_indices(["evtnum", "electrons_idx"])
    assert sv00.required_indices == ["evtnum", "electrons_idx"]
