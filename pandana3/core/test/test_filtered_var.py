from __future__ import annotations
import pytest
from typing import List, Set
import h5py as h5
import pandas as pd
import numpy as np
from pandana3.core.var import Var, SimpleVar, FilteredVar
from pandana3.core.cut import SimpleCut


def exercise_newly_constructed(
        var: Var,
        f: h5.File,
        expected_tablenames: Set[str],
        expected_result_columns: List[str],
) -> None:
    assert not var.prepared
    assert var.inq_tables_read() == expected_tablenames
    assert var.inq_result_columns() == expected_result_columns
    with pytest.raises(AssertionError):
        var.eval(f)


def exercise_inq_row_spec(var: Var, f: h5.File, expected_row_spec: List[str]) -> None:
    row_spec = var.inq_row_spec(f)
    assert row_spec == expected_row_spec


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


def test_fv00_row_spec(fv00: FilteredVar, datafile: h5.File) -> None:
    exercise_inq_row_spec(fv00, datafile, ["evtnum", "electrons_idx"])


def test_fv01_row_spec(fv01: FilteredVar, datafile: h5.File) -> None:
    exercise_inq_row_spec(fv01, datafile, ["evtnum", "electrons_idx"])


def test_fv02_row_spec(fv02: FilteredVar, datafile: h5.File) -> None:
    exercise_inq_row_spec(fv02, datafile, ["evtnum", "electrons_idx"])


def test_fv03_row_spec(fv03: FilteredVar, datafile: h5.File) -> None:
    exercise_inq_row_spec(fv03, datafile, ["evtnum", "electrons_idx", "hits_idx"])


def test_fv00_newly_constructed(fv00: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(fv00, dummyfile, {"electrons"}, ["pt", "eta", "rowid"])


def test_fv01_newly_constructed(fv01: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(fv01, dummyfile, {"events", "electrons"}, ["pt", "phi", "rowid"])


def test_fv02_newly_constructed(fv02: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(
        fv02, dummyfile, {"events", "electrons", "electrons_qual"}, ["q1", "q2", "rowid"]
    )


def test_fv03_newly_constructed(fv03: FilteredVar, dummyfile: h5.File) -> None:
    exercise_newly_constructed(
        fv03, dummyfile, {"events", "electrons", "electrons_hits"}, ["energy", "rowid"]
    )


def test_fv00_preparing(fv00: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(fv00, datafile, {"/electrons/pt", "/electrons/eta"})


def test_fv01_preparing(fv01: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(
        fv01,
        datafile,
        {
            "/electrons/pt",
            "/electrons/phi",
            "/electrons/evtnum",
            "/events/met",
            "/events/evtnum",
        },
    )


def test_fv02_preparing(fv02: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(
        fv02,
        datafile,
        {
            "/events/met",
            "/events/evtnum",
            "/electrons/pt",
            "/electrons/evtnum",
            "/electrons_qual/q1",
            "/electrons_qual/q2",
            "/electrons_qual/evtnum",
        },
    )


def test_fv03_preparing(fv03: FilteredVar, datafile: h5.File) -> None:
    exercise_preparing(
        fv03,
        datafile,
        {
            "/events/met",
            "/events/evtnum",
            "/electrons/pt",
            "/electrons/evtnum",
            "/electrons/electrons_idx",
            "/electrons_hits/energy",
            "/electrons_hits/evtnum",
            "/electrons_hits/electrons_idx",
        },
    )


def test_fv00_evaluating(fv00: FilteredVar, datafile: h5.File) -> None:
    fv00.prepare(datafile)
    df = fv00.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    # TODO: Consider whether we should only be obtaining a 'pt' column in the
    # dataframe that is returned by evaluting the FilteredVar.
    assert list(df.columns) == ["pt", "eta", "rowid"]
    assert len(df) == 18


def test_fv01_evaluating(fv01: FilteredVar, datafile: h5.File) -> None:
    fv01.prepare(datafile)
    df = fv01.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["pt", "phi", "rowid"]
    assert len(df) == 24


def test_fv02_evaluating(fv02: FilteredVar, datafile: h5.File) -> None:
    fv02.prepare(datafile)
    df = fv02.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns == ["q1", "q2"])
    assert len(df) == 24


def test_fv01_with_imposed_index(fv01: FilteredVar, datafile: h5.File) -> None:
    fv01.index_imposed = ["rowid"]
    fv01.prepare(datafile)
    assert fv01.prepared
    df = fv01.eval(datafile)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "rowid"


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
