import pytest
from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar, FilteredVar
from pandana3.core.cut import SimpleCut
from pandana3.core.index import SimpleIndex
from pandana3.core.grouping import Grouping
import h5py as h5
import pandas as pd
import numpy as np


def test_filtered_var_cut_and_var_use_same_var():
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


def test_filtered_var_compatible_cut_and_var():
    """Test a FilteredVar that applies a cut to one table that was
    calculated from another table."""
    # Make the 'base' Var for the Cut on electron quality.
    # Make the Cut for the FilteredVar
    # Make the 'base' Var for the FilteredVar: electron pt
    # Make the FilteredVar
    v1 = SimpleVar("electrons_qual", ["q1"])
    good = SimpleCut(v1, lambda df: df["q1"] > 0.75)
    v2 = SimpleVar("electrons", ["pt"])
    good_electrons = FilteredVar(v2, good)
    assert isinstance(good_electrons, FilteredVar)
    assert good_electrons.inq_datasets_read() == {"/electrons_qual/q1", "/electrons/pt"}
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


def test_filtered_var_incompatible_cut_and_var():
    # Should not be able to create a FilteredVar from a Cut with grouping level
    # of "electrons" and a Var with grouping level of "muons".
    electrons = SimpleVar("electrons", ["pt", "eta"])
    muons = SimpleVar("muons", ["pt", "phi"])
    bad_var = FilteredVar(electrons, SimpleCut(muons, lambda df: df["pt"] > 10.0))

    # TODO: Var.resolve_metadata should raise a better exception type than ValueError.
    # We need an exception type that carries information meaningful to the user about
    # what failed.
    with pytest.raises(ValueError):
        with h5.File("small.h5", "r") as f:
            bad_var.resolve_metadata(f)
            assert False  # should never get here, we should have raised an exception


def test_filtered_var_three():
    # Select events that are interesting (met > 10)
    # Select electron pt, eta that are in events that are interesting.
    events = SimpleVar("events", ["met"])
    good_events = SimpleCut(events, lambda df: df["met"] > 10.0)
    electrons = SimpleVar("electrons", ["pt", "eta"])
    good_electrons = FilteredVar(electrons, good_events)
    assert isinstance(good_electrons, FilteredVar)

    with h5.File("small.h5", "r") as f:
        column_names = good_electrons.resolve_metadata(f)
        assert good_electrons.inq_datasets_read() == {
            "/events/met",
            "/events/evtnum",
            "/electrons/pt",
            "/electrons/eta",
            "/electrons/evtnum",
        }

def test_doubly_filtered_var():
    # First select electron pt for electrons in events with met > 10
    # Use this for a cut on such electrons with pt > 15
    # Use this cut in a FilteredVar returning electons_qual q1, q2
    v1 = SimpleVar("events", ["met"])
    c1 = SimpleCut(v1, lambda df: df["met"] > 10)
    v2 = SimpleVar("electrons", ["pt"])
    fv1 = FilteredVar(v2, c1)

    c2 = SimpleCut(fv1, lambda df: df["pt"] > 15)
    v3 = SimpleVar("electrons_qual", ["q1", "q2"])
    fv2 = FilteredVar(v3, c2)

    assert isinstance(fv2, FilteredVar)

    with h5.File("small.h5", "r") as f:
        index_column_names = fv2.resolve_metadata(f)