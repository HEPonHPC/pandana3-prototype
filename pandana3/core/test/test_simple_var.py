import pytest
from pandana3.core.var import SimpleVar
import h5py as h5
import pandas as pd
import numpy as np

@pytest.fixture()
def sv00() -> SimpleVar:
    return SimpleVar("electrons", ["pt", "eta"])

def test_multiple_table_names():
    with pytest.raises(TypeError):
        _ = SimpleVar(["electrons", "muons"], ["pt"])


def test_column_names_not_list():
    with pytest.raises(TypeError):
        _ = SimpleVar("electrons", "pt")


def test_column_names_empy_list():
    with pytest.raises(ValueError):
        _ = SimpleVar("electrons", [])


def test_right_columns(sv00: SimpleVar) -> None:
    assert sv00.columns == ["pt", "eta"]
