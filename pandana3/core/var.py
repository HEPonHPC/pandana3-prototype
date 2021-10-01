"""Module Var provides the abstract class Var, and the concrete subclasses:
  ConstantVar: a Var representing a constant
  SimpleVar: a Var read directly from an HDF5 file table
  GroupedVar: a Var representing a grouping operation on a base Var
  MutatedVar: a Var representing a length-preserving transformation on a base Var
  FilteredVar: a Var representing the result of filtering another Var

  """
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set, Callable, Tuple, final
import pandas as pd
import h5py as h5
import numpy as np
from pandana3.core.cut import Cut

# IndexInfoType is the type returned from Var.resolve_metadata
IndexInfoType = Tuple[List[str], List[str]]


def common_prefix(bic: List[str], cic: List[str]) -> List[str]:
    """ "Return the subsequence of bic that is in cic. It is expected that the
    test for compatibility used by FilteredVar is already done."""
    return bic[0: len(cic)]


def verify_type(val, typ, msg: str) -> None:
    """If 'val' is not of type 'typ', raise a TypeError with the given message"""
    if not isinstance(val, typ):
        raise TypeError(msg)


class Var(ABC):
    """A Var is the basic representation of computation of data in PandAna.

    A Var is not, nor does it contain, a DataFrame. However, when given a file,
    it can be evaluated to yield a DataFrame."""

    def __init__(self):
        """The base class Var contains the data common to all Vars:
             prepared: the current state of the Var
             required_indices: the names of the index columns that are required
                  to evaluate this Var in the context in which it is used.
        """
        self.prepared = False
        self.required_indices: List[str] = []

    @final
    def prepare(self, f: h5.File) -> None:
        """Prepare for evaluation of this Var. This should be called directly by the
        user on the Var objects used directly in an analysis.

        This function is not to be overridden in derived classes."""
        assert not self.prepared
        self._do_prepare(f)
        self.set_prepared()

    @abstractmethod
    def _do_prepare(self, f: h5.File) -> None:
        """Do the preparation work necessary for this concrete Var type.

        This should not be called by any code outside of the Var base class."""
        raise NotImplementedError

    def set_prepared(self):
        """Record that this Var has been prepared.

        Derived classes should override this if they have special handling to do."""
        # TODO: Consider using the Template Method pattern here.
        self.prepared = True

    @final
    def set_required_indices(self, required_indices: List[str]) -> None:
        """Set the list of index columns that must be read during the
        evaluation of this Var.

        This method is not to be overridden in derived classes."""
        assert not self.prepared
        self._do_set_required_indices(required_indices)
        self.required_indices = required_indices

    @abstractmethod
    def _do_set_required_indices(self, required_indices: List[str]) -> None:
        raise NotImplementedError

    @final
    def inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read.

        This method is not to be overridden in derived classes."""
        assert self.prepared
        return self._do_inq_datasets_read()

    @abstractmethod
    def _do_inq_datasets_read(self) -> Set[str]:
        """"Return the full name of all datasets read when evaluating this var."""
        raise NotImplementedError

    @abstractmethod
    def inq_tables_read(self) -> Set[str]:
        """Return the names of the tables read"""
        raise NotImplementedError

    @abstractmethod
    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        raise NotImplementedError

    @final
    def eval(self, f: h5.File) -> pd.DataFrame:
        """Return the pd.DataFrame that is represented by this Var.

        This method should not be overridden by derived classes."""
        assert self.prepared
        assert f
        assert f.mode == "r"
        return self._do_eval(f)

    @abstractmethod
    def _do_eval(self, h5file: h5.File) -> pd.DataFrame:
        """Return the DataFrame represented by this Var.

        Derived classes implement this method to perform the action that is
        the defining behavior for that class."""
        raise NotImplementedError

    @abstractmethod
    def resolve_metadata(self, h5file: h5.File) -> IndexInfoType:
        """Return the index columns this Var might have, and the ones it appears to need.
        The second value is only correct for top-level Vars.

        Raise an exception if the Var is malformed."""
        raise NotImplementedError

    @final
    def filter_by(self, cut: Cut) -> FilteredVar:
        """Return a FilteredVar that uses self as a base, and applies
        the given Cut."""
        return FilteredVar(self, cut)


class ConstantVar(Var):
    """A ConstantVar evaluates to a DataFrame with a single column and a
    single row. It does not read any file.
    """

    def __init__(self, name: str, value: float):
        super().__init__()
        self.col_name = name
        self.value = value

    def _do_prepare(self, f: h5.File) -> None:
        """ConstantVar has no preparation to do."""
        pass

    def _do_set_required_indices(self, required_indices: List[str]) -> None:
        """ConstantVar has no indices to read."""
        pass

    def _do_inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read."""
        return set()

    def inq_tables_read(self) -> Set[str]:
        """Return a list of tables read"""
        return set()

    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        return [self.col_name]

    def _do_eval(self, h5file: h5.File) -> pd.DataFrame:
        """ConstantVars return a pd.DataFrame carrying the single value
        that was provided at construction time."""
        result = pd.DataFrame({self.col_name: np.array([self.value])})
        return result

    def resolve_metadata(self, h5file: h5.File) -> IndexInfoType:
        # ConstantVars do not have any metadata to resolve
        return [], []


class SimpleVar(Var):
    """A SimpleVar is a Var that is read directly from a file."""

    def __init__(self, table_name: str, column_names: List[str]):
        """Create a SimpleVar that will read the named columns from the named
        table.
        Invoke this like:
           myvar = SimpleVar("electrons", ["pt", "phi"])
        """
        verify_type(table_name, str, "table_name must be a string")
        verify_type(
            column_names, list, "column_names must be a nonempy list of strings"
        )
        if len(column_names) == 0:
            raise ValueError("column_names must be a nonempty list of strings")

        super().__init__()
        self.table = table_name
        self.columns = column_names

    def _do_prepare(self, f: h5.File) -> None:
        """A SimpleVar has no extra preparation to do."""
        pass

    def _do_set_required_indices(self, required_indices: List[str]) -> None:
        """SimpleVar has no work other than what the base does."""
        pass

    def _do_inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read."""
        physics_datasets = {f"/{self.table}/{col_name}" for col_name in self.columns}
        index_datasets = {
            f"/{self.table}/{col_name}" for col_name in self.required_indices
        }
        return physics_datasets.union(index_datasets)

    def inq_tables_read(self) -> Set[str]:
        """Return a list of tables read. For a SimpleVar, the length is always
        1."""
        return {self.table}

    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation. For a SimpleVar, this is always the same as the columns."""
        return self.columns

    def _do_eval(self, h5file: h5.File) -> pd.DataFrame:
        """Evaluate the Var by reading all the required data from the given
        h5.File object.

        In this first version, we have no limitation on the rows read; this
        always reads all rows."""

        # TODO: Replace this dictionary comprehension by something that will raise an
        # exception indicating which column(s) could not be found.
        try:
            all_column_names = self.columns + self.required_indices
            data = {col_name: h5file[f"/{self.table}/{col_name}"] for col_name in all_column_names}
        except KeyError as k:
            raise ValueError("Unable to find requested column in HDF5 file") from k
        result = pd.DataFrame(data)
        if len(self.required_indices) != 0:
            result.set_index(self.required_indices, inplace=True)
        return result

    def resolve_metadata(self, h5file: h5.File) -> Tuple[List[str], List[str]]:
        """Use the specified file f to fill out the metadata that can not
        be determined until access to the file is possible.
        """
        assert h5file, "Attempt to resolve Var metadata with a non-open File"
        index_columns = h5file[self.table].attrs["index_cols"].tolist()
        return index_columns, []


class GroupedVar(Var):
    """A GroupedVar has an underlying Var used in evaluation, and a grouping
    level that is used for reduction of the data.

      # Get sum of pts of electrons in each event.
      base = SimpleVar("electrons", ["pt"])
      reduction = np.sum
      myvar = GroupedVar(base, ["evtnum"], reduction)

    TODO: How do we get more than one column in a result? What is the equivalent
    of the tidyverse:

    electrons %>%
        group_by(evtnum) %>%
        summarize(nelec = n(), ptsum = sum(pt))

    which results in a dataframe with columns (nelec, ptsum)?
    """

    def __init__(
            self,
            var: Var,
            grouping: List[str],
            reduction: Callable[[np.ndarray], np.float64],
    ):
        super().__init__()
        self.var = var
        self.grouping = grouping
        self.reduction = reduction

    def _do_prepare(self, f: h5.File) -> None:
        return super().prepare(f)

    def _do_set_required_indices(self, required_indices: List[str]) -> None:
        return super()._do_set_required_indices(required_indices)

    def _do_inq_datasets_read(self) -> Set[str]:
        return self.var.inq_datasets_read()

    def inq_tables_read(self) -> Set[str]:
        return self.var.inq_tables_read()

    def inq_result_columns(self) -> List[str]:
        super().inq_result_columns()

    def _do_eval(self, h5file: h5.File) -> pd.DataFrame:
        temp = self.var.eval(h5file)
        # TODO: Is it more efficient to have the resulting dataframe carry the index?
        # TODO: Is it better to sort, or not to sort? Our data comes already sorted.
        return temp.groupby(self.grouping, as_index=False, sort=False).agg(
            self.reduction
        )
        # TODO: Some aggregation functions are directly callable on the grouped dataframe.
        # We may want special handling for them.

    def resolve_metadata(self, h5file: h5.File) -> IndexInfoType:
        return super().resolve_metadata(h5file)


class MutatedVar(Var):
    """A Mutated var applies a tranformation or mutation to another var.

    # Compute event distance to origin
    base = SimpleVar("electrons", ["x","y","z"])
    pythagoras = lambda df: np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    myvar = MutatedVarVar(base, pythagoras)

    TODO: The mutation is currently required to take and return a dataframe.
          What other call patterns should we support
    TODO: The example function is tame, but actually returns a pd.Series.
          Do we want to require pd.DataFrame?
    """

    def __init__(
            self, var: Var, name: str, mutation: Callable[[pd.DataFrame], pd.DataFrame]
    ):
        super().__init__()
        self.var = var
        self.name = name
        self.mutate = mutation

    def _do_prepare(self, f: h5.File) -> None:
        return super().prepare(f)

    def _do_set_required_indices(self, required_indices: List[str]) -> None:
        super()._do_set_required_indices(required_indices)

    def _do_inq_datasets_read(self) -> Set[str]:
        return self.var.inq_datasets_read()

    def inq_tables_read(self) -> Set[str]:
        return self.var.inq_tables_read()

    def inq_result_columns(self) -> List[str]:
        original_columns = self.var.inq_result_columns()
        return original_columns + [self.name]

    def _do_eval(self, h5file: h5.File) -> pd.DataFrame:
        temp = self.var.eval(h5file)

        # TODO: Do we want to return the computed frame or append the result
        # as a new column
        # TODO: this assumes the mutation yields a Series.
        temp[self.name] = self.mutate(temp)
        return temp

    def resolve_metadata(self, h5file: h5.File) -> IndexInfoType:
        return super().resolve_metadata(h5file)


class FilteredVar(Var):
    """A FilteredVar is the result of applying a cut to another Var."""

    def __init__(self, base: Var, cut: Cut):
        super().__init__()
        assert isinstance(base, Var)
        assert isinstance(cut, Cut)
        self.base = base
        self.cut = cut

    @staticmethod
    def determine_required(bic: List[str], cic: List[str]) -> List[str]:
        """Given two lists of index column names,"""
        if bic == cic:
            return []
        return common_prefix(bic, cic)

    def _do_prepare(self, f: h5.File) -> None:
        _, self.required_indices = self.resolve_metadata(f)
        self.base.set_required_indices(self.required_indices)
        self.cut.set_required_indices(self.required_indices)

    def set_prepared(self) -> None:
        self.base.set_prepared()
        self.cut.set_prepared()
        self.prepared = True

    def _do_set_required_indices(self, required_indices: List[str]) -> None:
        self.base.set_required_indices(required_indices)
        self.cut.set_required_indices(required_indices)

    def _do_inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read."""
        all_datasets = set.union(
            self.base.inq_datasets_read(), self.cut.inq_datasets_read()
        )
        # We want to remove any duplicates from the list. Note that this does not preserve
        # order; we should not care about the order.
        return set(all_datasets)

    def inq_tables_read(self) -> Set[str]:
        """Return a set of tables read"""
        return self.base.inq_tables_read() | self.cut.inq_tables_read()

    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        return self.base.inq_result_columns()

    def _do_eval(self, h5file: h5.File) -> pd.DataFrame:
        # TODO: Optimize this so that we don't evaluate
        # the vars involved more than once each.
        tmp = self.base.eval(h5file)
        good = self.cut.eval(h5file)
        return tmp[good]

    def add_columns(self, column_names: List[str]) -> None:
        """Add a new columns to be read."""
        raise NotImplementedError("We don't know how to add columns to a FilteredVar")

    def resolve_metadata(self, h5file: h5.File) -> IndexInfoType:
        assert h5.File, "Attempt to resolve Var metadata with a non-open File"
        base_all, _ = self.base.resolve_metadata(h5file)
        cut_all, _ = self.cut.resolve_metadata(h5file)

        if not self.check_compatible(base_all, cut_all):
            raise ValueError("FilteredVar has incompatible index columns")

        apparently_required = self.determine_required(base_all, cut_all)
        return base_all, apparently_required

    @staticmethod
    def check_compatible(
            base_index_columns: List[str], cut_index_columns: List[str]
    ) -> bool:
        if base_index_columns == cut_index_columns:
            return True

        if len(cut_index_columns) > len(base_index_columns):
            return False

        for b, c in zip(base_index_columns, cut_index_columns):
            if b != c:
                return False
        return True
