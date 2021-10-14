import pytest
from pandana3.core.var import Var


class BadVar(Var):
    """This class is defective because it does not implement the required
    methods."""

    pass


def test_var_subclass_requires_eval():
    """A Var subclass that does not implement the Var protocol should not be importable."""
    with pytest.raises(TypeError):
        _ = BadVar()
