"""Module index defines the several types of Index required by
pandana3.
"""
from __future__ import annotations
from pandana3.core.grouping import Grouping

# TODO: What interface does Index need?


class Index:
    def __init__(self, trivial: bool = True, grouping: Grouping = Grouping()):
        self.is_trivial = trivial
        self.gping = grouping

    def grouping(self) -> Grouping:
        return self.gping


class SimpleIndex(Index):
    def __init__(self, trivial: bool = True):
        super(SimpleIndex, self).__init__(trivial)


class MultiIndex(Index):
    def __init__(self, trivial: bool = True, grouping: Grouping = Grouping()):
        super(MultiIndex, self).__init__(trivial, grouping)


def make_index(idx1: Index, idx2: Index) -> Index:
    is_trivial = idx1.is_trivial and idx2.is_trivial
    if isinstance(idx1, SimpleIndex) and isinstance(idx2, SimpleIndex):
        return SimpleIndex(is_trivial)
    return MultiIndex(is_trivial, idx1.grouping().combine(idx2.grouping()))
