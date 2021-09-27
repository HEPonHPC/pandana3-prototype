from __future__ import annotations
import pytest
import h5py as h5
import numpy as np
from pandana3.core.var import SimpleVar, FilteredVar
from pandana3.core.cut import SimpleCut


@pytest.fixture(scope="session")
def dummyfile() -> h5.File:
    # Note: we open in write mode, so that there doesn't need to be a file
    # on the filesystem, and with backing_store=False, so that we don't
    # create one when the h5.File is closed.
    f = h5.File("dummy.h5", mode="w", driver="core", backing_store=False)
    yield f
    f.close()


@pytest.fixture(scope="session")
def datafile() -> h5.File:
    f = h5.File("small.h5", "r")
    yield f
    f.close()


@pytest.fixture()
def sv00() -> SimpleVar:
    return SimpleVar("electrons", ["pt", "eta"])


@pytest.fixture()
def fv00() -> FilteredVar:
    base = SimpleVar("electrons", ["pt", "eta"])
    cut = SimpleCut(base, lambda ele: np.abs(ele.eta) < 1.5)
    return FilteredVar(base, cut)


@pytest.fixture()
def fv01() -> FilteredVar:
    cut = SimpleCut(SimpleVar("events", ["met"]),
                    lambda df: df["met"] > 10.0)
    return FilteredVar(SimpleVar("electrons", ["pt", "phi"]),
                       cut)


@pytest.fixture()
def fv02() -> FilteredVar:
    v1 = SimpleVar("events", ["met"])
    c1 = SimpleCut(v1, lambda df: df["met"] > 10.0)
    v2 = SimpleVar("electrons", ["pt"])
    fv1 = FilteredVar(v2, c1)
    cut = SimpleCut(fv1, lambda df: df["pt"] > 15.0)
    var = SimpleVar("electrons_qual", ["q1", "q2"])
    return FilteredVar(var, cut)


@pytest.fixture()
def fv03() -> FilteredVar:
    v1 = SimpleVar("events", ["met"])
    c1 = SimpleCut(v1, lambda df: df["met"] > 10.0)
    v2 = SimpleVar("electrons", ["pt"])
    fv1 = FilteredVar(v2, c1)
    cut = SimpleCut(fv1, lambda df: df["pt"] > 15.0)
    var = SimpleVar("electrons_hits", ["energy"])
    return FilteredVar(var, cut)
