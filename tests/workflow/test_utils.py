from __future__ import annotations

import pytest
import numpy as np

from cellpose_kit.workflow.utils import get_axis, count_channels


def test_get_axis_returns_index():
    assert get_axis("TCZYX", "C") == 1
    assert get_axis("TCZYX", "Z") == 2
    assert get_axis("YXC", "C") == 2
    assert get_axis("YX", "T") is None


def test_count_channels_no_channel_axis():
    img = np.zeros((10, 10), dtype=np.uint8)
    assert count_channels(img, "YX") == 1


def test_count_channels_with_channel_axis():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert count_channels(img, "YXC") == 3


def test_count_channels_multidim():
    img = np.zeros((5, 2, 10, 10), dtype=np.uint8)
    assert count_channels(img, "TCYX") == 2


def test_count_channels_validates_axis_order():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        count_channels(img, "YX")  # wrong number of axes
