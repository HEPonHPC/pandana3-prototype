"""Module index defines the several types of Index required by
pandana3.
"""
from __future__ import annotations
from pandana3.core.grouping import Grouping

# TODO: What interface does Index need?


class Index:
    """ "Index is the base class for indices.

    Each concrete index is either a SimpleIndex or a MultiIndex. Each type
    of index may or may not be trivial.

    An index is trivial if it has an entry for every row of its associated
    table. In such cases it is often not necessary to actually read any
    indexing information from the file.
    """

    def __init__(self, trivial: bool = True, grouping: Grouping = Grouping()):
        self.is_trivial = trivial
        self.gping = grouping

    def grouping(self) -> Grouping:
        """Return the grouping level associated with this index."""
        return self.gping


class SimpleIndex(Index):
    """A SimpleIndex represents an index that contains row numbers, not values from
    actual data columns.
    """

    def __init__(self, trivial: bool = True):
        super().__init__(trivial)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleIndex):
            return NotImplemented
        return self.is_trivial == other.is_trivial


class MultiIndex(Index):
    """A MultiIndex represents and index that contains data from columns."""

    def __init__(self, trivial: bool = True, grouping: Grouping = Grouping()):
        super().__init__(trivial, grouping)


def make_index(idx1: Index, idx2: Index) -> Index:
    """Create an Index that is a combination of idx1 and idx2."""
    is_trivial = idx1.is_trivial and idx2.is_trivial
    if isinstance(idx1, SimpleIndex) and isinstance(idx2, SimpleIndex):
        return SimpleIndex(is_trivial)
    return MultiIndex(is_trivial, idx1.grouping().combine(idx2.grouping()))


def __eq__(self, other: object) -> bool:
    if not isinstance(object, MultiIndex):
        return NotImplemented
    return self.is_trivial == other.is_trivial and self.gping == other.gping
