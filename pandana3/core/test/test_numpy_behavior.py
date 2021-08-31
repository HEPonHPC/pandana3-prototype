from __future__ import annotations

import itertools
import numpy as np
import pytest


@pytest.fixture()
def num_events() -> int:
    return 3


@pytest.fixture()
def electron_multiplicities() -> List[int]:
    return [2, 0, 3]


@pytest.fixture()
def hit_multiplicities() -> List[int]:
    return [5, 4, 3, 2, 1]


def test_hit_index_generation(num_events, hit_multiplicities):
    a = [range(n) for n in hit_multiplicities]
    hit_idx = np.array(list(itertools.chain(*a)))
    assert isinstance(hit_idx, np.ndarray)
    assert len(hit_idx) == num_events * len(hit_multiplicities)
    assert np.array_equal(
        hit_idx, np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0])
    )


# Start working on this test
def test_electron_index_generation(
    num_events, electron_multiplicities, hit_multiplicities
):
    a = [range(n) for n in hit_multiplicities]
    electrons_idx = np.array(list(itertools.chain(*a)))
    assert isinstance(electrons_idx, np.ndarray)
    assert len(electrons_idx) == num_events * len(hit_multiplicities)
    assert np.array_equal(
        electrons_idx, np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2])
    )
