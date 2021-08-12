import pytest
from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar, FilteredVar
from pandana3.core.cut import SimpleCut
from pandana3.core.index import SimpleIndex, MultiIndex
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
    assert x.table == "electrons"
    assert x.columns == ["evtnum", "pt", "phi"]
    assert x.inq_tables_read() == ["electrons"]
    assert x.inq_result_columns() == x.columns
    assert x.inq_datasets_read() == [
        "/electrons/evtnum",
        "/electrons/pt",
        "/electrons/phi",
    ]
    idx = x.inq_index()
    assert idx.is_trivial, "SimpleVar did not return a trivial Index"

    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 29
        assert (d.keys() == ["evtnum", "pt", "phi"]).all()


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
    assert set(x.inq_datasets_read()) == {"/electrons/pt", "/electrons/evtnum"}
    # assert x.inq_index() == base.inq_index()

    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 9
        assert (d.keys() == ["evtnum", "pt"]).all()
        assert np.abs(d["pt"][2] - 82.386965) < 1.0e-3


def test_mutated_var_basic():
    base = SimpleVar("electrons", ["x", "y", "z"])
    mutation = lambda df: np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    x = MutatedVar(base, "dist", mutation)
    assert x is not None
    assert x.inq_tables_read() == ["electrons"]
    assert set(x.inq_datasets_read()) == {
        "/electrons/x",
        "/electrons/y",
        "/electrons/z",
    }
    assert set(x.inq_result_columns()) == {"x", "y", "z", "dist"}
    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        print(d)
        assert isinstance(d, pd.DataFrame)
        assert (d.keys() == ["x", "y", "z", "dist"]).all()
        assert np.abs(d["dist"][2] - 0.750797) < 1.0e-3


def test_filtered_var_basic():
    """Test a FilteredVar that applies a cut to the same table from which
    the cut was calculated."""
    base = SimpleVar("electrons", ["pt", "eta"])
    central = lambda ele: np.abs(ele.eta) < 1.5
    cut = SimpleCut(base, central)
    x = FilteredVar(base, cut)
    assert x is not None
    assert set(x.inq_datasets_read()) == {"/electrons/pt", "/electrons/eta"}
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
    pass
