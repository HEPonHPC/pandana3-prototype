import h5py as h5


class Cut:
    pass


class SimpleCut(Cut):
    def __init__(self, base, predicate):
        self.base = base
        self.predicate = predicate

    def eval(self, f: h5.File):
        """Return a bool series."""
        full = self.base.eval(f)
        return self.predicate(full)
