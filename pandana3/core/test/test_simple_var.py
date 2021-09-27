import pytest
import h5py as h5
import pandas as pd

from pandana3.core.var import SimpleVar


@pytest.fixture()
def sv00() -> SimpleVar:
    return SimpleVar("electrons", ["pt", "eta"])


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
    with pytest.raises(AssertionError):
        _ = sv00.eval(dummyfile)


def test_newly_constructed_is_unprepared(sv00: SimpleVar) -> None:
    assert not sv00.prepared


def test_newly_constructed_indices(sv00: SimpleVar) -> None:
    assert len(sv00.required_indices) == 0


def test_sv00_newly_constructed(sv00: SimpleVar, dummyfile: h5.File) -> None:
    assert not sv00.prepared
    assert sv00.inq_tables_read() == ["electrons"]
    assert sv00.inq_result_columns() == ["pt", "eta"]
    with pytest.raises(AssertionError):
        sv00.eval(dummyfile)


def test_newly_constructed_datasets_read(sv00: SimpleVar) -> None:
    with pytest.raises(AssertionError):
        assert sv00.inq_datasets_read()


def test_prepare_sets_state(sv00: SimpleVar, datafile: h5.File) -> None:
    sv00.prepare(datafile)
    assert sv00.prepared


def test_datasets_read(sv00: SimpleVar, datafile: h5.File) -> None:
    sv00.prepare(datafile)
    assert sv00.inq_datasets_read() == {"/electrons/pt", "/electrons/eta"}


def test_eval_no_index(sv00: SimpleVar, datafile: h5.File) -> None:
    sv00.prepare(datafile)
    df = sv00.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["pt", "eta"]
    assert isinstance(df.index, pd.RangeIndex)
    assert len(df) == 29


def test_set_required_indices(sv00: SimpleVar) -> None:
    sv00.set_required_indices(["evtnum", "electrons_idx"])
    assert sv00.required_indices == ["evtnum", "electrons_idx"]


def test_resolve_metadata(sv00: SimpleVar, datafile: h5.File) -> None:
    sv00.prepare(datafile)
    available, needed = sv00.resolve_metadata(datafile)
    assert available == ["evtnum", "electrons_idx"]
    assert needed == []
