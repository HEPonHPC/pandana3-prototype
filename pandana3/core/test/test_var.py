import pytest
from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar, FilteredVar
from pandana3.core.cut import SimpleCut
from pandana3.core.index import SimpleIndex
from pandana3.core.grouping import Grouping
import h5py as h5
import pandas as pd
import numpy as np


class BadVar(Var):
    """This class is defective because it does not implement the required
    methods."""

    pass


def test_var_subclass_requires_eval():
    """A Var subclass that does not implement the Var protocol should not be importable."""
    with pytest.raises(TypeError):
        _ = BadVar()
