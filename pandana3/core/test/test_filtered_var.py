from __future__ import annotations
import pytest
from typing import List, Set
import h5py as h5
import pandas as pd
import numpy as np
from pandana3.core.var import Var, SimpleVar, FilteredVar
from pandana3.core.cut import SimpleCut
from pandana3.core.index import SimpleIndex
from pandana3.core.grouping import Grouping


def exercise_newly_constructed(var: Var, f: h5.File, expected_tablenames: Set[str],
                               expected_result_columns: List[str]) -> None:
    assert not var.prepared
    assert var.inq_tables_read() == expected_tablenames
    assert var.inq_result_columns() == expected_result_columns
    with pytest.raises(AssertionError):
        var.eval(f)


def exercise_preparing(var: Var, f: h5.File, expected_datasets_read: Set[str]) -> None:
    assert not var.prepared
    var.prepare(f)
    assert var.prepared
    datasets_read = var.inq_datasets_read()
    assert datasets_read == expected_datasets_read


def test_check_compatible():
    assert FilteredVar.check_compatible(["a", "b"], ["a"])
    assert FilteredVar.check_compatible([], [])
    assert FilteredVar.check_compatible(["a"], ["a"])
    assert not FilteredVar.check_compatible(["a"], ["b"])
    assert not FilteredVar.check_compatible(["a"], ["a", "b"])


def test_fv00_newly_constructed(fv00: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(fv00, dummyfile, {"electrons"}, ["pt", "eta"])


def test_fv01_newly_constructed(fv01: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(fv01, dummyfile, {"events", "electrons"}, ["pt", "phi"])


def test_fv02_newly_constructed(fv02: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(fv02, dummyfile, {"events", "electrons", "electrons_qual"},
                               ["q1", "q2"])


def test_fv03_newly_constructed(fv03: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(fv03, dummyfile, {"events", "electrons", "electrons_hits"},
                               ["energy"])


def test_fv00_preparing(fv00: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(fv00, datafile, {"/electrons/pt", "/electrons/eta"})


def test_fv01_preparing(fv01: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(fv01, datafile,
                       {"/electrons/pt", "/electrons/phi", "/electrons/evtnum", "/events/met", "/events/evtnum"})


def test_fv02_preparing(fv02: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(fv02, datafile,
                       {"/events/met", "/events/evtnum", "/electrons/pt", "/electrons/evtnum",
                        "/electrons_qual/q1", "/electrons_qual/q2", "/electrons_qual/evtnum"})


def test_fv03_preparing(fv03: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(fv03, datafile,
                       {"/events/met", "/events/evtnum", "/electrons/pt", "/electrons/evtnum",
                        "/electrons/electrons_idx",
                        "/electrons_hits/energy", "/electrons_hits/evtnum", "/electrons_hits/electrons_idx"})


def test_fv00_evaluating(fv00: FilteredVar, datafile: h5.File) -> None:
    fv00.prepare(datafile)
    df = fv00.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    # TODO: Consider whether we should only be obtaining a 'pt' column in the
    # dataframe that is returned by evaluting the FilteredVar.
    assert list(df.columns) == ["pt", "eta"]
    assert len(df) == 18


def test_fv01_evaluating(fv01: FilteredVar, datafile: h5.File) -> None:
    fv01.prepare(datafile)
    df = fv01.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["pt", "phi"]
    assert len(df) == 24


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
    assert events.required_indices == []
    good_events = SimpleCut(events, lambda df: df["met"] > 10.0)
    electrons = SimpleVar("electrons", ["pt", "eta"])
    good_electrons = FilteredVar(electrons, good_events)
    assert isinstance(good_electrons, FilteredVar)

    with h5.File("small.h5", "r") as f:
        good_electrons.prepare(f)
        assert good_electrons.required_indices == ["evtnum"]
        assert events.required_indices == ["evtnum"]

        column_names = good_electrons.resolve_metadata(f)
        assert column_names == ["evtnum", "electrons_idx"]
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
