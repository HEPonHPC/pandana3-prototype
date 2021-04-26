class Grouping:
    def __init__(self, column_names=None):
        """Create a Grouping.

        column_names is the list of the names of columns to be used in a group;
        if column_names is none, this is the flag that "all" columns are to be
        used, and each row is its own group; such a Grouping is trivial."""
        # TODO: Is it better to have self.columns always be a list? Then
        # is_trivial would have to test for an empty list, but maybe other
        # code would be simpler. Wait to see what use of Grouping looks like
        # to decide if this change should be made
        self.columns = column_names

    def is_trivial(self):
        """A Grouping is trivial if each row is its own group."""
        return self.columns is None
