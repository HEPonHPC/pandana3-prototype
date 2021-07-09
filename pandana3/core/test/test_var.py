from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar
import h5py as h5
import pandas as pd
import numpy as np


class NoEvalVar(Var):
    """This class is defective because it does not implement the required
    method `eval`."""

    pass


def test_var_subclass_requires_eval():
    try:
        _ = NoEvalVar()
        assert False, "failed to throw required exception"
    except TypeError:
        pass
    except:
        assert False, "threw the wrong kind of exception"


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


def test_simple_var_construction():
    try:
        _ = SimpleVar(["electrons", "muons"], ["pt"])
        assert False, "Failed to throw required exception"
    except TypeError:
        pass
    except:
        assert False, "threw the wrong kind of exception"
    try:
        _ = SimpleVar("electrons", "pt")
        assert False, "Failed to throw required exception"
    except TypeError:
        pass
    except:
        assert False, "threw the wrong kind of exception"

    try:
        _ = SimpleVar("electrons", [])
        assert False, "Failed to throw required exception"
    except ValueError:
        pass
    except:
        assert False, "threw the wrong kind of exception"


def test_grouped_var_basic():
    base = SimpleVar("electrons", ["pt"])
    x = GroupedVar(base, ["evtnum"], np.sum)
    assert x is not None
    assert x.inq_tables_read() == ["electrons"]
    assert set(x.inq_datasets_read()) == set(["/electrons/pt", "/electrons/evtnum"])
    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 9
        assert (d.keys() == ["evtnum", "pt"]).all()
        assert np.abs(d["pt"][2] - 82.386965) < 1.0e-3


def test_grouped_var_duplicated_columns():
    base = SimpleVar("electrons", ["pt"])


def test_mutated_var_basic():
    base = SimpleVar("electrons", ["x", "y", "z"])
    mutation = lambda df: np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    x = MutatedVar(base, "dist", mutation)
    assert x is not None
    assert x.inq_tables_read() == ["electrons"]
    assert set(x.inq_datasets_read()) == set(
        ["/electrons/x", "/electrons/y", "/electrons/z"]
    )
    assert set(x.inq_result_columns()) == set(["x", "y", "z", "dist"])
    with h5.File("small.h5", "r") as f:
        d = x.eval(f)
        print(d)
        assert isinstance(d, pd.DataFrame)
        assert (d.keys() == ["x", "y", "z", "dist"]).all()
        assert np.abs(d["dist"][2] - 0.750797) < 1.0e-3
