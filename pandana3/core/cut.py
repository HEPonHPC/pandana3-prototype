from __future__ import annotations
import h5py as h5
from pandana3.core.index import Index
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Set, Callable


class Cut(ABC):
    @abstractmethod
    def eval(self, f: h5.File) -> pd.Series:
        pass

    @abstractmethod
    def inq_datasets_read(self) -> Set[str]:
        """Return the full names of the datasets that will be read by this cut."""
        pass

    @abstractmethod
    def inq_tables_read(self) -> List[str]:
        """Return the names of tables that will be read by this cut."""
        pass

    @abstractmethod
    def inq_index(self) -> Index:
        pass


class SimpleCut(Cut):
    def __init__(self, base, predicate: Callable[[pd.DataFrame], pd.Series]):
        """base must be a Var, and predicate is a callable object that will
        be passed a dataframe and must return a boolean series.

        mycut = SimpleCut(myvar, lambda df:df["q1"]>0.75)
        """
        # TODO: Is there a better way to get type checking on 'base' to work?
        # It would be nice to have the type specficiation on the argument list
        # but that introduces a circular dependency between cut.py and var.py.
        from pandana3.core.var import Var

        self.base: Var = base
        self.predicate = predicate

    def inq_datasets_read(self) -> Set[str]:
        return self.base.inq_datasets_read()

    def inq_tables_read(self) -> List[str]:
        return self.base.inq_tables_read()

    def inq_index(self) -> Index:
        """The index for a SimpleCut is not trivial. It has the same type as the
        index for the underlying Var."""
        idx = self.base.inq_index()
        idx.is_trivial = False
        return idx

    def eval(self, f: h5.File) -> pd.Series:
        """Return a bool series."""
        full = self.base.eval(f)
        return self.predicate(full)
