"""Module cut provides the abstract base class Cut and the concrete subclasses:

  SimpleCut: represents a cut based upon applying a stateless predicate to the contents of a Var.

"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Set, Callable, Tuple, final
import h5py as h5
import pandas as pd


class Cut(ABC):
    def __init__(self):
        self.prepared = False

    @final
    def prepare(self, f: h5.File) -> None:
        """Prepare for evaluation of this Var. This should be called directly by the
        user on the Var objects used directly in an analysis.

        This method is not to be overridden in derived classes."""
        assert not self.prepared
        self._do_prepare(f)
        self.prepared = True

    @abstractmethod
    def _do_prepare(self, f: h5.File) -> None:
        raise NotImplementedError

    def set_prepared(self) -> None:
        """Record this this Cut has been prepared.

        Derived classes should override this if they have special processing to do."""
        self.prepared = True

    @abstractmethod
    def eval(self, f: h5.File) -> pd.Series:
        """ " Evaluate the Cut using data supplied b the given file f, returning a
        Pandas Series of True/False values.
        """
        raise NotImplementedError

    @final
    def inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read.

        This method is not to be overridden in derived classes."""
        assert self.prepared
        return self._do_inq_datasets_read()

    @abstractmethod
    def _do_inq_datasets_read(self) -> Set[str]:
        """Return the full names of the datasets that will be read by this cut."""
        raise NotImplementedError

    @abstractmethod
    def inq_tables_read(self) -> Set[str]:
        """Return the names of tables that will be read by this cut."""
        raise NotImplementedError


    @abstractmethod
    def resolve_metadata(self, h5file: h5.File) -> Tuple[List[str], List[str]]:
        return [], []

    @abstractmethod
    def set_required_indices(self, required_indices: List[str]) -> None:
        raise NotImplementedError


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

        super().__init__()

        self.base: Var = base
        self.predicate = predicate

    def _do_prepare(self, f: h5.File) -> None:
        self.base.prepare(f)

    def set_prepared(self) -> None:
        self.base.set_prepared()
        self.prepared = True

    def _do_inq_datasets_read(self) -> Set[str]:
        """"A SimpleCut reads the datasets from it's contained Var."""
        return self.base.inq_datasets_read()

    def inq_tables_read(self) -> Set[str]:
        return self.base.inq_tables_read()


    def eval(self, f: h5.File) -> pd.Series:
        """Return a bool series."""
        full = self.base.eval(f)
        return self.predicate(full)

    def resolve_metadata(self, h5file: h5.File) -> Tuple[List[str], List[str]]:
        return self.base.resolve_metadata(h5file)

    def set_required_indices(self, required_indices: List[str]) -> None:
        self.base.set_required_indices(required_indices)
