from __future__ import annotations

import numpy as np

from cellpose_kit.workflow.utils import get_axis, count_axis


def test_get_axis_returns_index():
    assert get_axis("TCZYX", "C") == 1
    assert get_axis("TCZYX", "Z") == 2
    assert get_axis("YXC", "C") == 2
    assert get_axis("YX", "T") is None


def test_count_axis_no_channel_axis_returns_one():
    img = np.zeros((10, 10), dtype=np.uint8)
    assert count_axis(img, "YX", "C") == 1


def test_count_axis_with_channel_axis():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert count_axis(img, "YXC", "C") == 3


def test_count_axis_multidim():
    img = np.zeros((5, 2, 10, 10), dtype=np.uint8)
    assert count_axis(img, "TCYX", "C") == 2


def test_count_axis_missing_axis_returns_one():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert count_axis(img, "YX", "C") == 1
