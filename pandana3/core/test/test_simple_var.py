import pytest
from pandana3.core.var import SimpleVar, GroupedVar, Var, MutatedVar, FilteredVar
from pandana3.core.cut import SimpleCut
from pandana3.core.index import SimpleIndex
from pandana3.core.grouping import Grouping
import h5py as h5
import pandas as pd
import numpy as np


def test_simple_var_multiple_table_names():
    with pytest.raises(TypeError):
        _ = SimpleVar(["electrons", "muons"], ["pt"])


def test_simple_var_column_names_not_list():
    with pytest.raises(TypeError):
        _ = SimpleVar("electrons", "pt")


def test_simple_var_column_names_empy_list():
    with pytest.raises(ValueError):
        _ = SimpleVar("electrons", [])
