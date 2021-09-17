from __future__ import annotations

import pytest
import numpy as np

from tools.numpy_numerology import (
    repeated_indices,
    make_outer_indices,
    flattened_ranges,
)

# ELECTRONS TABLE
# event_idx electron_idx
# -----------------------
#  0         0
#  0         1
# ------------------------
# ------------------------
#  2         0
#  2         1
#  2         2
# ------------------------

# ------------------------
# ELECTRONS_HITS TABLE
# event_idx electron_idx hit_idx
# ---------------------------
# 0          0             0
# 0          0             1
# ............................
# 0          1             0
# 0          1             1
# 0          1             2
# 0          1             3
# ----------------------------
# ----------------------------
# 2          0             0
# 2          0             1
# 2          0             2
# ............................
# 2          1             0
# 2          1             1
# ............................
# 2          2             0
# ---------------------------


@pytest.fixture()
def num_events() -> int:
    return 3


@pytest.fixture()
def electrons_per_event() -> List[int]:
    return [2, 0, 3]


@pytest.fixture()
def hits_per_electron() -> List[int]:
    return [2, 4, 3, 2, 1]


def test_repeated_indices():
    assert repeated_indices([2, 0, 3]) == [0, 0, 2, 2, 2]


def test_make_outer_indices():
    assert make_outer_indices([1, 2], [2, 1, 2]) == [0, 0, 0, 1, 1]


def test_hit_index_generation(hits_per_electron):
    hit_idx = np.array(flattened_ranges(hits_per_electron))
    assert isinstance(hit_idx, np.ndarray)
    num_hits = len(hit_idx)
    assert num_hits == sum(hits_per_electron)
    assert np.array_equal(hit_idx, np.array([0, 1, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0]))


def test_electron_index_generation(num_events, electrons_per_event, hits_per_electron):
    electrons_idx = np.array(make_outer_indices(electrons_per_event, hits_per_electron))
    assert isinstance(electrons_idx, np.ndarray)
    assert len(electrons_idx) == sum(hits_per_electron)
    assert np.array_equal(electrons_idx, np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2]))
