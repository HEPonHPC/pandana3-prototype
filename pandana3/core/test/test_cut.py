import pytest
import pandas as pd
import h5py as h5
from pandana3.core.cut import SimpleCut
from pandana3.core.var import SimpleVar
from pandana3.core.index import SimpleIndex

def test_simple_cut():
    electron_pt = SimpleVar("electrons", ["pt"])
    c1 = SimpleCut(electron_pt, lambda pt: pt>50)
    assert c1.inq_tables_read() == ["electrons"]
    assert not c1.inq_index().is_trivial # Because we have applied a cut!
    assert isinstance(c1.inq_index(), SimpleIndex)
    assert c1.inq_datasets_read() == {"/electrons/pt"}

    with h5.File("small.h5", "r") as f:
        vals = c1.eval(f)
        assert isinstance(vals, pd.Series)

