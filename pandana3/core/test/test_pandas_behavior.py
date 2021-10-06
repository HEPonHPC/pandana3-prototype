# This file is for tests of expected behavior of pandas itself, rather than
# tests for parts of pandana.
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest


@pytest.fixture()
def events() -> pd.DataFrame:
    """A DataFrame with three events"""
    # evtnum      met     nelectrons
    # 0           15.0    2
    # 1           20.0    0
    # 2            9.0    3
    d = {
        "evtnum": np.array([0, 1, 2]),
        "met": np.array([15.0, 20.0, 9.0]),
        "nelectrons": np.array([2, 0, 3]),
    }
    return pd.DataFrame(d)


@pytest.fixture()
def electrons() -> pd.DataFrame:
    """A DataFrame with kinematic data for 5 electrons in three events."""
    #  evtnum   electrons_idx   pt    nhits
    #  0        0               10.0  1
    #  0        1               20.0  3
    #  2        0               30.0  1
    #  2        1               40.0  2
    #  2        2               50.0  1
    d = {
        "evtnum": np.array([0, 0, 2, 2, 2]),
        "electrons_idx": np.array([0, 1, 0, 1, 2]),
        "pt": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    }
    return pd.DataFrame(d)


@pytest.fixture()
def electrons_pt(electrons) -> pd.DataFrame:
    """Selection from electrons: DataFrame with only 'pt' column."""
    return electrons[["pt"]]


@pytest.fixture()
def electrons_qual():
    """A DataFrame with quality data for 5 electrons in three events."""
    #  evtnum   electrons_idx   q1
    #  0        0               0.9
    #  0        1               0.1
    #  2        0               0.8
    #  2        1               0.4
    #  2        2               0.5
    d = {
        "evtnum": np.array([0, 0, 2, 2, 2]),
        "electrons_idx": np.array([0, 1, 0, 1, 2]),
        "q1": np.array([0.9, 0.1, 0.8, 0.4, 0.5]),
    }
    return pd.DataFrame(d)


@pytest.fixture()
def electrons_hits(electrons):
    """A DataFrame with hit information for 5 electrons in 3 events."""
    #  evtnum   electrons_idx  hits_idx energy
    #  0        0              0        1.0
    #  0        1              0        2.0
    #  0        1              1        3.0
    #  0        1              2        4.0
    #  2        0              0        5.0
    #  2        1              0        6.0
    #  2        1              1        7.0
    #  2        2              0        8.0
    d = {
        "evtnum": np.array([0, 0, 0, 0, 2, 2, 2, 2]),
        "electrons_idx": np.array([0, 1, 1, 1, 0, 1, 1, 2]),
        "hits_idx": np.array([0, 0, 1, 2, 0, 0, 1, 0]),
        "energy": np.array([1., 2., 3., 4., 5., 6., 7., 8.]),
    }
    return pd.DataFrame(d)


def test_pull_column_from_dataframe(electrons):
    """ "We can extract a single column from a DataFrame, yielding a Series."""
    pt = electrons["pt"]
    assert isinstance(pt, pd.Series)
    assert pt.dtype == np.float64
    assert pt.name == "pt"


def test_select_one_column_from_dataframe(electrons):
    """We can select a DataFrame with fewer columns from a DataFrame"""
    pt = electrons[["pt"]]
    assert isinstance(pt, pd.DataFrame)
    assert list(pt.columns) == ["pt"]


def test_align_congruent_dataframes_without_multiindex(
        electrons_pt: pd.DataFrame, electrons_qual: pd.DataFrame
):
    good = electrons_qual["q1"] > 0.75  # boolean series
    assert np.array_equal(good, np.array([True, False, True, False, False]))
    good_pt = electrons_pt[good]  # filtered dataframe
    assert list(good_pt.columns) == ["pt"]
    assert np.array_equal(good_pt["pt"], np.array([10.0, 30.0]))
    assert np.array_equal(good_pt.index, np.array([0, 2]))  # simple index
