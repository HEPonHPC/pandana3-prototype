from __future__ import annotations
import pytest
import h5py as h5
from pandana3.core.var import SimpleVar


@pytest.fixture(scope="session")
def dummyfile() -> h5.File:
    # Note: we open in write mode, so that there doesn't need to be a file
    # on the filesystem, and with backing_store=False, so that we don't
    # create one when the h5.File is closed.
    f = h5.File("dummy.h5", mode="w", driver="core", backing_store=False)
    yield f
    f.close()


@pytest.fixture(scope="session")
def datafile() -> h5.File:
    f = h5.File("small.h5", "r")
    yield f
    f.close()


@pytest.fixture()
def sv00() -> SimpleVar:
    return SimpleVar("electrons", ["pt", "eta"])
