import pytest
from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar, FilteredVar
from pandana3.core.cut import SimpleCut
from pandana3.core.index import SimpleIndex
from pandana3.core.grouping import Grouping
import h5py as h5
import pandas as pd
import numpy as np


def test_grouped_var_basic():
    base = SimpleVar("electrons", ["pt"])
    x = GroupedVar(base, ["evtnum"], np.sum)
    assert x is not None
    assert x.inq_tables_read() == ["electrons"]
    assert x.inq_datasets_read() == {"/electrons/pt", "/electrons/evtnum"}
    assert x.inq_index() == base.inq_index()

    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 9
        assert list(d.columns) == ["evtnum", "pt"]
        assert np.abs(d["pt"][2] - 82.386965) < 1.0e-3
