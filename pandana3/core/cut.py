import h5py as h5
from abc import ABC, abstractmethod


class Cut(ABC):
    @abstractmethod
    def eval(self, h5file):
        pass

    @abstractmethod
    def inq_datasets_read(self):
        """Return the full names of the datasets that will be read by this cut."""
        pass

    @abstractmethod
    def inq_tables_read(self):
        """Return the names of tables that will be read by this cut."""
        pass


class SimpleCut(Cut):
    def __init__(self, base, predicate):
        self.base = base
        self.predicate = predicate

    def inq_datasets_read(self):
        return self.base.inq_datasets_read()

    def inq_tables_read(self):
        return self.base.inq_tables_read()

    def inq_index(self):
        pass

    def eval(self, f: h5.File):
        """Return a bool series."""
        full = self.base.eval(f)
        return self.predicate(full)
