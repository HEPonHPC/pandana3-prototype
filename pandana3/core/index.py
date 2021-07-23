class Index:
    def __init__(self, trivial=True):
        self.is_trivial = trivial


# TODO: Do we need SimpleIndex and MultiIndex? If so, how do they differ and
# what is the interface to be supported by the base class Index? Should Index
# be an abstract class?


class SimpleIndex(Index):
    def __init__(self, trivial=True):
        super(SimpleIndex, self).__init__(trivial)


class MultiIndex(Index):
    def __init__(self, trivial=True):
        super(MultiIndex, self).__init__(trivial)
