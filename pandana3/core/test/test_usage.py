"""These tests verify the desired usage of components of PandAna."""

from pandana3.core.var import SimpleVar, MutatedVar
import h5py as h5
import pandas as pd


# Exercise the creation of various types of Vars.
pt = SimpleVar("electrons", ["pt"])
vertex = SimpleVar("electrons", ["x", "y", "z"])


with h5.File("small.h5", "r") as f:
    d1 = pt.eval(f)
    assert isinstance(d1, pd.DataFrame)
    assert len(d1) == 29

    d2 = vertex.eval(f)

