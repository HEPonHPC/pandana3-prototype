import h5py as h5
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Set


class Cut(ABC):
    @abstractmethod
    def eval(self, f: h5.File):
        pass

    @abstractmethod
    def inq_datasets_read(self) -> Set[str]:
        """Return the full names of the datasets that will be read by this cut."""
        pass

    @abstractmethod
    def inq_tables_read(self) -> List[str]:
        """Return the names of tables that will be read by this cut."""
        pass


class SimpleCut(Cut):
    def __init__(self, base, predicate):
        self.base = base
        self.predicate = predicate

    def inq_datasets_read(self) -> Set[str]:
        return self.base.inq_datasets_read()

    def inq_tables_read(self) -> List[str]:
        return self.base.inq_tables_read()

    def inq_index(self):
        """The index for a SimpleCut is not trivial. It has the same type as the
        index for the underlying Var."""
        idx = self.base.inq_index()
        idx.is_trivial = False
        return idx

    def eval(self, f: h5.File) -> pd.Series:
        """Return a bool series."""
        full = self.base.eval(f)
        return self.predicate(full)
