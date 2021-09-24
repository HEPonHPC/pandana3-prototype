"""Module Var provides the abstract class Var, and the concrete subclasses:
  ConstantVar: a Var representing a constant
  SimpleVar: a Var read directly from an HDF5 file table
  GroupedVar: a Var representing a grouping operation on a base Var
  MutatedVar: a Var representing a length-preserving transformation on a base Var
  FilteredVar: a Var representing the result of filtering another Var

  """
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set, Callable
import pandas as pd
import h5py as h5
import numpy as np
from pandana3.core.grouping import Grouping
from pandana3.core import index
from pandana3.core.index import Index, SimpleIndex
from pandana3.core.cut import Cut


def verify_type(val, typ, msg: str) -> None:
    """If 'val' is not of type 'typ', raise a TypeError with the given message"""
    if not isinstance(val, typ):
        raise TypeError(msg)


class Var(ABC):
    """A Var is the basic representation of computation of data in PandAna.

    A Var is not, nor does it contain, a DataFrame. However, when given a file,
    it can be evaluated to yield a DataFrame."""

    @abstractmethod
    def prepare(self, f: h5.File) -> None:
        """Prepare for evaluation of this Var. This should be called directly by the
        user on the Var objects used directly in an analysis."""
        raise NotImplementedError

    @abstractmethod
    def set_required_indices(self, required_indices: List[str]) -> None:
        """Set the list of index columns that must be read during the
        evaluation of this Var."""
        raise NotImplementedError

    @abstractmethod
    def inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read."""
        raise NotImplementedError

    @abstractmethod
    def inq_tables_read(self) -> List[str]:
        """Return a list of tables read"""
        raise NotImplementedError

    @abstractmethod
    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        raise NotImplementedError

    @abstractmethod
    def inq_index(self) -> Index:
        """Return the Index to be used for this Var."""
        raise NotImplementedError

    @abstractmethod
    def inq_grouping(self) -> Grouping:
        """Return the Grouping used for this Var."""
        raise NotImplementedError

    @abstractmethod
    def eval(self, h5file: h5.File) -> pd.DataFrame:
        """Return the DataFrame represented by this Var."""
        raise NotImplementedError

    @abstractmethod
    def add_columns(self, column_names: List[str]) -> None:
        """Add one or more new columns to be read."""
        raise NotImplementedError

    @abstractmethod
    def resolve_metadata(self, h5file: h5.File) -> List[str]:
        """Return the index columns this Var might have.

        Raise an exception if the Var is malformed."""
        raise NotImplementedError

    def filter_by(self, cut: Cut) -> FilteredVar:
        """Return a FilteredVar that uses self as a base, and applies
        the given cut."""
        return FilteredVar(self, cut)


class ConstantVar(Var):
    """A ConstantVar evaluates to a DataFrame with a single column and a
    single row. It does not read any file.
    """

    def __init__(self, name: str, value: float):
        self.col_name = name
        self.value = value

    def prepare(self, f: h5.file) -> None:
        """ConstantVar has no preparation to do."""
        pass

    def set_required_indices(self, required_indices: List[str]) -> None:
        """ConstantVar has no indices to read."""
        pass

    def inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read."""
        return set()

    def inq_tables_read(self) -> List[str]:
        """Return a list of tables read"""
        return []

    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        return [self.col_name]

    def inq_index(self) -> Index:
        """Return the Index to be used for this Var."""
        return SimpleIndex()

    def inq_grouping(self) -> Grouping:
        """Return the Grouping used for this Var."""
        return Grouping()

    def eval(self, h5file: h5.File) -> pd.DataFrame:
        result = pd.DataFrame({self.col_name: np.array([self.value])})
        return result

    def add_columns(self, column_names: List[str]):
        """Add a new columns to be read."""
        raise TypeError("you can't add columns to a ConstVar")

    def resolve_metadata(self, h5file: h5.File) -> List[str]:
        # ConstantVars do not have any metadata to resolve
        return []


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
        self.table = table_name
        self.columns = column_names
        self.index_columns_to_read = []

    def prepare(self, f: h5.File) -> None:
        """A SimpleVar has no preparation to do."""
        pass

    def set_required_indices(self, required_indices: List[str]) -> None:
        self.index_columns_to_read = required_indices

    def inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read."""
        physics_datasets = {f"/{self.table}/{col_name}" for col_name in self.columns}
        index_datasets = {
            f"/{self.table}/{col_name}" for col_name in self.index_columns_to_read
        }
        return physics_datasets.update(index_datasets)

    def inq_tables_read(self) -> List[str]:
        """Return a list of tables read. For a SimpleVar, the length is always
        1."""
        return [self.table]

    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation. For a SimpleVar, this is always the same as the columns."""
        return self.columns

    def inq_index(self) -> Index:
        """Return the Index to be used for this Var. For a SimpleVar, this is
        always a trivial SimpleIndex, because we're reading all the rows."""
        return SimpleIndex()

    def inq_grouping(self) -> Grouping:
        """Return the Grouping used for this Var. A SimpleVar is ungrouped,
        and so returns a Grouping that is fundamental."""
        return Grouping(column_names=None)

    def eval(self, h5file: h5.File) -> pd.DataFrame:
        """Evaluate the Var by reading all the required data from the given
        h5.File object.

        In this first version, we have no limitation on the rows read; this
        always reads all rows."""
        assert h5file, "Attempt to evaluate a Var with a non-open File"
        # TODO: Replace this dictionary comprehension by something that will raise an
        # exception indicating which column(s) could not be found.
        try:
            data = {name: h5file[f"/{self.table}/{name}"] for name in self.columns}
        except KeyError as k:
            raise ValueError("Unable to find requested column in HDF5 file") from k
        result = pd.DataFrame(data)
        return result

    def add_columns(self, column_names: List[str]) -> None:
        """Add a new columns to be read."""
        if not isinstance(column_names, list):
            raise TypeError("column_names must be a nonempty list of strings")
        if len(column_names) == 0:
            raise ValueError("column_names must be a nonempty list of strings")
        # TODO: There must be a more efficient way to do this addition.
        for name in column_names:
            if not name in self.columns:
                self.columns.append(name)

    def resolve_metadata(self, h5file: h5.File) -> List[str]:
        """Use the specified file f to fill out the metadata that can not
        be determined until access to the file is possible.
        """
        assert h5file, "Attempt to resolve Var metadata with a non-open File"
        self.index_columns = h5file[self.table].attrs["index_cols"].tolist()
        return self.index_columns


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
        self.var = var
        self.var.add_columns(grouping)
        self.grouping = Grouping(grouping)
        self.reduction = reduction

    def prepare(self, f: h5.File) -> None:
        return super().prepare(f)

    def set_required_indices(self, required_indices: List[str]) -> None:
        return super().set_required_indices(required_indices)

    def inq_datasets_read(self) -> Set[str]:
        return self.var.inq_datasets_read()

    def inq_tables_read(self) -> List[str]:
        return self.var.inq_tables_read()

    def inq_result_columns(self) -> List[str]:
        raise NotImplementedError("GroupedVar inq_result_columns is not implemented.")

    def inq_index(self) -> Index:
        return self.var.inq_index()

    def inq_grouping(self) -> Grouping:
        return self.grouping

    def eval(self, h5file: h5.File) -> pd.DataFrame:
        temp = self.var.eval(h5file)
        # TODO: Is it more efficient to have the resulting dataframe carry the index?
        # TODO: Is it better to sort, or not to sort? Our data comes already sorted.
        return temp.groupby(self.grouping.column_names, as_index=False, sort=False).agg(
            self.reduction
        )
        # TODO: Some aggregation functions are directly callable on the grouped dataframe.
        # We may want special handling for them.

    def add_columns(self, column_names: List[str]) -> None:
        self.var.add_columns(column_names)

    def resolve_metadata(self, h5file: h5.File) -> List[str]:
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
        self.var = var
        self.name = name
        self.mutate = mutation

    def prepare(self, f: h5.File) -> None:
        return super().prepare(f)

    def set_required_indices(self, required_indices: List[str]) -> None:
        return super().set_required_indices(required_indices)

    def inq_datasets_read(self) -> Set[str]:
        return self.var.inq_datasets_read()

    def inq_tables_read(self) -> List[str]:
        return self.var.inq_tables_read()

    def inq_result_columns(self) -> List[str]:
        original_columns = self.var.inq_result_columns()
        return original_columns + [self.name]

    def inq_index(self) -> Index:
        return self.var.inq_index()

    def inq_grouping(self) -> Grouping:
        return self.var.inq_grouping()

    def add_columns(self, column_names: List[str]) -> None:
        self.var.add_columns(column_names)

    def eval(self, h5file: h5.File) -> pd.DataFrame:
        temp = self.var.eval(h5file)

        # TODO: Do we want to return the computed frame or append the result
        # as a new column
        # TODO: this assumes the mutation yields a Series.
        temp[self.name] = self.mutate(temp)
        return temp

    def resolve_metadata(self, h5file: h5.File) -> List[str]:
        return super().resolve_metadata(h5file)


class FilteredVar(Var):
    """A FilteredVar is the result of applying a cut to another Var."""

    def __init__(self, base: Var, cut: Cut):
        assert isinstance(base, Var)
        assert isinstance(cut, Cut)
        self.base = base
        self.cut = cut
        self.required_index_columns = None

    def prepare(self, f: h5.File) -> None:
        assert !self.prepared

        self.resolve_metadata(f)  # We are intentionally ignoring the returned value
        self.base.set_required_indices(self.required_index_columns)
        self.cut.set_required_indices(self.required_index_columns)
        self.set_prepared()

    def set_prepared(self) -> None:
        self.base.set_prepared()
        self.cut.set_prepared()
        self.prepared = True

    def set_required_indices(self, required_indices: List[str]) -> None:
        self.required_index_columns = required_indices

    def inq_datasets_read(self) -> Set[str]:
        """Return the (full) names of the datasets to be read."""
        all_datasets = set.union(
            self.base.inq_datasets_read(), self.cut.inq_datasets_read()
        )
        # We want to remove any duplicates from the list. Note that this does not preserve
        # order; we should not care about the order.
        return set(all_datasets)

    def inq_tables_read(self) -> List[str]:
        """Return a list of tables read"""
        all_tables = self.base.inq_tables_read() + self.cut.inq_tables_read()
        # We want to remove any duplicates from the list. Note that this does not preserve
        # order; we should not care about the order.
        return list(set(all_tables))

    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        return self.base.inq_result_columns()

    def inq_index(self) -> Index:
        """Return the Index to be used for this Var."""
        cut_idx = self.cut.inq_index()
        base_idx = self.base.inq_index()
        return index.make_index(base_idx, cut_idx)

    def inq_grouping(self) -> Grouping:
        """Return the Grouping used for this Var."""
        return self.base.inq_grouping()

    def eval(self, h5file: h5.File) -> pd.DataFrame:
        # TODO: Optimize this so that we don't evaluate
        # the vars involved more than once each.
        tmp = self.base.eval(h5file)
        good = self.cut.eval(h5file)
        return tmp[good]

    def add_columns(self, column_names: List[str]) -> None:
        """Add a new columns to be read."""
        raise NotImplementedError("We don't know how to add columns to a FilteredVar")

    def resolve_metadata(self, h5file: h5.File) -> List[str]:
        assert h5.File, "Attempt to resolve Var metadata with a non-open File"
        base_index_columns = self.base.resolve_metadata(h5file)
        cut_index_columns = self.cut.resolve_metadata(h5file)

        if base_index_columns == cut_index_columns:
            # We don't have to read any index columns to evaluate this
            # var.
            return base_index_columns

        if len(cut_index_columns) > len(base_index_columns):
            # TODO: use a custom exception type with meaningful data here
            raise ValueError("FilteredVar has incompatible index columns")

        common_prefix = []
        for b, c in zip(base_index_columns, cut_index_columns):
            if b != c:
                raise ValueError("FilteredVar has incompatible index columns")
            common_prefix.append(b)

        self.required_index_columns = common_prefix
        return base_index_columns
