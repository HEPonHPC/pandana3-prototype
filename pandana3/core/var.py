from pandana3.core.index import Index
from pandana3.core.grouping import Grouping
from abc import ABC, abstractmethod
import pandas as pd


class Var(ABC):
    """A Var is the basic representation of data in PandAna."""

    @abstractmethod
    def eval(self, h5file):
        pass


class SimpleVar(Var):
    """A SimpleVar is a Var that is read directly from a file."""

    def __init__(self, table_name, column_names):
        """Create a SimpleVar that will read the named columns from the named
        table."""
        self.table = table_name
        self.columns = column_names

    def get_datasets_read(self):
        """Return the (full) names of the datasets to be read."""
        return [f"/{self.table}/{col_name}" for col_name in self.columns]

    def get_tables_read(self):
        """Return a list of tables read. For a SimpleVar, the length is always
        1."""
        return [self.table]

    def get_result_columns(self):
        """Return the column names in the DataFrame that will be the result of
        evaluation. For a SimpleVar, this is always the same as the columns."""
        return self.columns

    def get_index(self):
        """Return the Index to be used for this Var. For a SimpleVar, this is
        always a trivial index, because we're reading all the rows."""
        return Index(trivial=True)

    def get_grouping(self):
        """Return the Grouping used for this Var. A SimpleVar is ungrouped,
        and so returns a Grouping that is fundamental."""
        return Grouping(column_names=None)

    def eval(self, h5file):
        """Evaluate the Var by reading all the required data from the given
        h5.File object.

        In this first version, we have no limitation on the rows read; this
        always reads all rows."""
        assert h5file, "Attempt to evaluate a Var with a non-open File"
        dsets = [h5file[name] for name in self.get_datasets_read()]
        data = {name: vals for (name, vals) in zip(self.columns, dsets)}
        return pd.DataFrame(data)
