import pytest
from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar, FilteredVar
from pandana3.core.cut import SimpleCut
import h5py as h5
import pandas as pd
import numpy as np


def test_mutated_var_basic():
    base = SimpleVar("electrons", ["x", "y", "z"])
    x = MutatedVar(
        base, "dist", lambda df: np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    )
    assert x is not None
    assert x.inq_tables_read() == {"electrons"}
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
