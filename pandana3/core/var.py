from .index import Index
from pandana3.core.grouping import Grouping
from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class Var(ABC):
    """A Var is the basic representation of data in PandAna."""

    @abstractmethod
    def inq_datasets_read(self):
        """Return the (full) names of the datasets to be read."""
        pass

    @abstractmethod
    def inq_tables_read(self):
        """Return a list of tables read"""
        pass

    @abstractmethod
    def inq_result_columns(self):
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        pass

    @abstractmethod
    def inq_index(self):
        """Return the Index to be used for this Var."""
        pass

    @abstractmethod
    def inq_grouping(self):
        """Return the Grouping used for this Var."""
        pass

    @abstractmethod
    def eval(self, h5file):
        pass

    @abstractmethod
    def add_columns(self, column_names: List[str]):
        """Add a new columns to be read."""
        pass


class ConstantVar(Var):
    """A ConstantVar evaluates to a DataFrame with a single column and a
    single row. It does not read any file.
    """

    def __init__(self, name: str, value):
        self.value = pd.DataFrame({name: value})

    def inq_datasets_read(self) -> List[str]:
        """Return the (full) names of the datasets to be read."""
        return []

    @abstractmethod
    def inq_tables_read(self) -> List[str]:
        """Return a list of tables read"""
        pass

    @abstractmethod
    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation."""
        pass

    @abstractmethod
    def inq_index(self):
        """Return the Index to be used for this Var."""
        pass

    @abstractmethod
    def inq_grouping(self) -> Grouping:
        """Return the Grouping used for this Var."""
        pass

    @abstractmethod
    def eval(self, h5file):
        pass

    @abstractmethod
    def add_columns(self, column_names: List[str]):
        """Add a new columns to be read."""
        pass

    def inq_datasets_read(self):
        return []

    def eval(self, _):
        return self.value


class SimpleVar(Var):
    """A SimpleVar is a Var that is read directly from a file."""

    def __init__(self, table_name: str, column_names: List[str]):
        """Create a SimpleVar that will read the named columns from the named
        table.
        Invoke this like:
           myvar = SimpleVar("electrons", ["pt", "phi"])
        """
        if not isinstance(table_name, str):
            raise TypeError("table_name for SimpleVar must be a string")
        if not isinstance(column_names, list):
            raise TypeError(
                "column_names for SimpleVar must be a nonempty list of strings"
            )
        if len(column_names) == 0:
            raise ValueError(
                "column_names for SimpleVar must be a nonempty list of strings"
            )
        self.table = table_name
        self.columns = column_names

    def inq_datasets_read(self) -> List[str]:
        """Return the (full) names of the datasets to be read."""
        return [f"/{self.table}/{col_name}" for col_name in self.columns]

    def inq_tables_read(self) -> List[str]:
        """Return a list of tables read. For a SimpleVar, the length is always
        1."""
        return [self.table]

    def inq_result_columns(self) -> List[str]:
        """Return the column names in the DataFrame that will be the result of
        evaluation. For a SimpleVar, this is always the same as the columns."""
        return self.columns

    def inq_index(self):
        """Return the Index to be used for this Var. For a SimpleVar, this is
        always a trivial index, because we're reading all the rows."""
        return Index(trivial=True)

    def inq_grouping(self):
        """Return the Grouping used for this Var. A SimpleVar is ungrouped,
        and so returns a Grouping that is fundamental."""
        return Grouping(column_names=None)

    def eval(self, h5file):
        """Evaluate the Var by reading all the required data from the given
        h5.File object.

        In this first version, we have no limitation on the rows read; this
        always reads all rows."""
        assert h5file, "Attempt to evaluate a Var with a non-open File"
        dsets = [h5file[name] for name in self.inq_datasets_read()]
        data = {name: vals for (name, vals) in zip(self.columns, dsets)}
        return pd.DataFrame(data)

    def add_columns(self, column_names: List[str]) -> None:
        """Add a new columns to be read."""
        if not isinstance(column_names, list):
            raise TypeError("column_names must be a nonempty list of strings")
        if len(column_names) == 0:
            raise ValueError("column_names must be a nonempty list of strings")
        self.columns.extend(column_names)


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

    def __init__(self, var, grouping: List[str], reduction):
        self.var = var
        self.var.add_columns(grouping)
        self.grouping = Grouping(grouping)
        self.reduction = reduction

    def inq_datasets_read(self) -> List[str]:
        return self.var.inq_datasets_read()

    def inq_tables_read(self) -> List[str]:
        return self.var.inq_tables_read()

    def inq_result_columns(self) -> List[str]:
        raise NotImplementedError("GroupedVar inq_result_columns is not implemented.")

    def inq_index(self):
        return self.var.inq_index()

    def inq_grouping(self) -> Grouping:
        return self.grouping

    def eval(self, h5file):
        temp = self.var.eval(h5file)
        # TODO: Is it more efficient to have the resulting dataframe carry the index?
        # TODO: Is it better to sort, or not to sort? Our data comes already sorted.
        return temp.groupby(self.grouping.column_names, as_index=False, sort=False).agg(
            self.reduction
        )
        # TODO: Some aggregation functions are directly callable on the grouped dataframe.
        # We may want special handling for them.

    def add_columns(self, column_names) -> None:
        self.var.add_columns(column_names)


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

    def __init__(self, var, name, mutation):
        self.var = var
        self.name = name
        self.mutate = mutation

    def inq_datasets_read(self) -> List[str]:
        return self.var.inq_datasets_read()

    def inq_tables_read(self) -> List[str]:
        return self.var.inq_tables_read()

    def inq_result_columns(self) -> List[str]:
        original_columns = self.var.inq_result_columns()
        return original_columns + [self.name]

    def inq_index(self):
        return self.var.inq_index()

    def inq_grouping(self) -> Grouping:
        return self.var.inq_grouping()

    def add_columns(self, column_names) -> None:
        self.var.add_columns(column_names)

    def eval(self, h5file):
        temp = self.var.eval(h5file)

        # TODO: Do we want to return the computed frame or append the result
        # as a new column
        # TODO: this assumes the mutation yields a Series.
        temp[self.name] = self.mutate(temp)
        return temp
