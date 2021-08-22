import pytest
from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar, FilteredVar
from pandana3.core.cut import SimpleCut
from pandana3.core.index import SimpleIndex
from pandana3.core.grouping import Grouping
import h5py as h5
import pandas as pd
import numpy as np


class BadVar(Var):
    """This class is defective because it does not implement the required
    method `eval`."""

    pass


def test_var_subclass_requires_eval():
    """A Var subclass that does not implement the Var protocol should not be importable."""
    with pytest.raises(TypeError):
        _ = BadVar()


def test_simple_var_basic():
    x = SimpleVar("electrons", ["evtnum", "pt", "phi"])
    assert x is not None
    assert isinstance(x, SimpleVar)
    assert x.table == "electrons"
    assert x.columns == ["evtnum", "pt", "phi"]
    assert x.inq_tables_read() == ["electrons"]
    assert x.inq_result_columns() == x.columns
    assert x.inq_datasets_read() == {
        "/electrons/evtnum",
        "/electrons/pt",
        "/electrons/phi",
    }
    idx = x.inq_index()
    assert idx.is_trivial, "SimpleVar did not return a trivial Index"
    assert x.index_columns is None

    with h5.File("small.h5", "r") as f:
        x.resolve_metadata(f)
        assert x.index_columns == ["evtnum", "electrons_idx"]
        d = x.eval(f)
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 29
        assert list(d.columns) == ["evtnum", "pt", "phi"]


def test_simple_var_multiple_table_names():
    with pytest.raises(TypeError):
        _ = SimpleVar(["electrons", "muons"], ["pt"])


def test_simple_var_column_names_not_list():
    with pytest.raises(TypeError):
        _ = SimpleVar("electrons", "pt")


def test_simple_var_column_names_empy_list():
    with pytest.raises(ValueError):
        _ = SimpleVar("electrons", [])


def test_grouped_var_basic():
    base = SimpleVar("electrons", ["pt"])
    x = GroupedVar(base, ["evtnum"], np.sum)
    assert x is not None
    assert x.inq_tables_read() == ["electrons"]
    assert x.inq_datasets_read() == {"/electrons/pt", "/electrons/evtnum"}
    # assert x.inq_index() == base.inq_index()

    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 9
        assert list(d.columns) == ["evtnum", "pt"]
        assert np.abs(d["pt"][2] - 82.386965) < 1.0e-3


def test_mutated_var_basic():
    base = SimpleVar("electrons", ["x", "y", "z"])
    x = MutatedVar(
        base, "dist", lambda df: np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    )
    assert x is not None
    assert x.inq_tables_read() == ["electrons"]
    assert x.inq_datasets_read() == {
        "/electrons/x",
        "/electrons/y",
        "/electrons/z",
    }
    assert set(x.inq_result_columns()) == {"x", "y", "z", "dist"}
    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        assert isinstance(d, pd.DataFrame)
        assert list(d.columns) == ["x", "y", "z", "dist"]
        assert np.abs(d["dist"][2] - 0.750797) < 1.0e-3


def test_filtered_var_basic():
    """Test a FilteredVar that applies a cut to the same table from which
    the cut was calculated."""
    # TODO: Consider whether we should only be obtaining a 'pt' column in the
    # dataframe that is returned by evaluting the FilteredVar.
    base = SimpleVar("electrons", ["pt", "eta"])
    cut = SimpleCut(base, lambda ele: np.abs(ele.eta) < 1.5)
    x = FilteredVar(base, cut)
    assert x is not None
    assert x.inq_datasets_read() == {"/electrons/pt", "/electrons/eta"}
    assert len(x.inq_datasets_read()) == 2
    assert x.inq_tables_read() == ["electrons"]
    assert set(x.inq_result_columns()) == {"pt", "eta"}
    idx = x.inq_index()
    assert idx is not None
    assert isinstance(idx, SimpleIndex)
    assert not idx.is_trivial

    var2 = base.filter_by(cut)

    with h5.File("small.h5", "r") as f:
        cut_df = x.eval(f)
        assert isinstance(cut_df, pd.DataFrame)

        cut_series = cut.eval(f)
        base_df = base.eval(f)
        cut_df2 = base_df[cut_series]
        cut_df3 = var2.eval(f)

        assert cut_df.equals(cut_df2)
        assert cut_df.equals(cut_df3)


def test_filtered_var_two():
    """Test a FilteredVar that applies a cut to one table that was
    calculated from another table."""
    # Make the 'base' Var for the Cut on electron quality.
    # Make the Cut for the FilteredVar
    # Make the 'base' Var for the FilteredVar: electron pt
    # Make the FilteredVar
    v1 = SimpleVar("elequal", ["q1"])
    good = SimpleCut(v1, lambda df: df["q1"] > 0.75)
    v2 = SimpleVar("electrons", ["pt"])
    good_electrons = FilteredVar(v2, good)
    assert isinstance(good_electrons, FilteredVar)
    assert good_electrons.inq_datasets_read() == {"/elequal/q1", "/electrons/pt"}
    assert good_electrons.inq_result_columns() == ["pt"]
    assert not good_electrons.inq_index().is_trivial
    assert isinstance(good_electrons.inq_index(), SimpleIndex)
    assert good_electrons.inq_grouping() == Grouping()
    assert good_electrons.inq_grouping().is_trivial

    with h5.File("small.h5", "r") as f:
        df = good_electrons.eval(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 9  # there are 9 good electrons
        assert list(df.columns) == ["pt"]
        expected = np.array([11.6, 30.8, 11.7, 7.34, 44.3, 29.1, 13.9, 58.7, 34.4])
        assert np.array(df["pt"]) == pytest.approx(expected, rel=0.01)
        assert np.array_equal(
            df.index.values, np.array([0, 10, 11, 12, 18, 20, 21, 25, 28])
        )


def test_filtered_var_bad():
    # Should not be able to create a FilteredVar from a Cut with grouping level
    # of "electrons" and a Var with grouping level of "events" or "muons".
    electrons = SimpleVar("electrons", ["pt", "eta"])
    muons = SimpleVar("muons", ["pt", "phi"])

    with pytest.raises(ValueError):
        bad_var = FilteredVar(electrons, SimpleCut(muons, lambda df: df["pt"] > 10.0))


def test_filtered_var_three():
    # Select events that are interesting (met > 10)
    # Select electron pt, eta that are in events that are interesting.
    events = SimpleVar("events", ["met"])
    good_events = SimpleCut(events, lambda df: df["met"] > 10.0)
    electrons = SimpleVar("electrons", ["pt", "eta"])
    good_electrons = FilteredVar(electrons, good_events)
    assert isinstance(good_electrons, FilteredVar)

    with h5.File("small.h5", "r") as f:
        good_electrons.resolve_metadata(f)
        assert good_electrons.inq_datasets_read() == {
            "/events/met",
            "/events/eventum",
            "/electrons/pt",
            "/electrons/eta",
            "/electrons/eventnum",
        }
