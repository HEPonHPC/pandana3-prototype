from typing import Optional, List
import copy


class Grouping:
    """A Grouping defines the aggregation level of a Var; in other words,
    it defines the nature of the object that is summarized by a row in
    the DataFrame to which that Var will evaluate.
    """

    def __init__(self, column_names: Optional[List[str]] = None):
        """Create a Grouping.

        column_names is the list of the names of columns to be used in a group;
        if column_names is none, this is the flag that "all" columns are to be
        used, and each row is its own group. Such a Grouping is trivial.
        """
        self.column_names = column_names if column_names is not None else []

    def is_trivial(self) -> bool:
        """A Grouping is trivial if each row is its own group."""
        return len(self.column_names) == 0

    def combine(self, other: "Grouping") -> "Grouping":
        """Return a new Grouping that is the result of combining 'other'
        with self.
        """
        if self.is_trivial():
            return copy.deepcopy(other)
        if other.is_trivial():
            return copy.deepcopy(self)
        my_column_names = set(self.column_names)
        other_column_names = set(other.column_names)
        if my_column_names <= other_column_names:
            return copy.deepcopy(other)
        if other_column_names < my_column_names:
            return copy.deepcopy(self)
        raise ValueError("Can not combine incompatible Groupings")

    def __eq__(self, other) -> bool:
        """Two Groupings are equal if they have the same column names."""
        # TODO: this implementation cares about the order of the names.
        # Should we instead *not* care about the order?
        return self.column_names == other.column_names
