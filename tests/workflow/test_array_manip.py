from __future__ import annotations

import pytest
import numpy as np

from cellpose_kit.workflow.array_manip import pad_to_3_channels, split_channels, get_frames_from_array


def test_pad_to_3_channels_success():
    img = np.ones((10, 10, 2), dtype=np.uint8)
    result = pad_to_3_channels(img, "YXC")
    
    assert result.shape == (10, 10, 3)
    assert np.all(result[:, :, :2] == 1)
    assert np.all(result[:, :, 2] == 0)


def test_pad_to_3_channels_no_channel_axis():
    img = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="no 'C' axis"):
        pad_to_3_channels(img, "YX")


def test_pad_to_3_channels_wrong_channel_count():
    img = np.ones((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly 2 input channels"):
        pad_to_3_channels(img, "YXC")


def test_pad_to_3_channels_channel_at_different_position():
    img = np.ones((2, 10, 10), dtype=np.uint8)
    result = pad_to_3_channels(img, "CYX")
    
    assert result.shape == (3, 10, 10)
    assert np.all(result[:2, :, :] == 1)
    assert np.all(result[2, :, :] == 0)


def test_split_channels_success():
    img = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)  # 1x2x3 (YXC)
    arrays, axis_order = split_channels(img, "YXC")
    
    assert len(arrays) == 3
    assert axis_order == "YX"
    assert arrays[0].shape == (1, 2)
    assert np.array_equal(arrays[0], [[1, 4]])
    assert np.array_equal(arrays[1], [[2, 5]])
    assert np.array_equal(arrays[2], [[3, 6]])


def test_split_channels_no_channel_axis():
    img = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="no 'C' axis"):
        split_channels(img, "YX")


def test_split_channels_at_different_position():
    img = np.ones((2, 10, 10), dtype=np.uint8)
    arrays, axis_order = split_channels(img, "CYX")
    
    assert len(arrays) == 2
    assert axis_order == "YX"
    assert arrays[0].shape == (10, 10)


def test_get_frames_from_array_no_time_axis():
    img = np.ones((10, 10), dtype=np.uint8)
    frames = get_frames_from_array(img, "YX")
    
    assert len(frames) == 1
    assert np.array_equal(frames[0], img)


def test_get_frames_from_array_with_time_axis():
    img = np.arange(60).reshape(3, 4, 5).astype(np.uint8)  # TYX
    frames = get_frames_from_array(img, "TYX")
    
    assert len(frames) == 3
    assert frames[0].shape == (4, 5)
    assert np.array_equal(frames[0], img[0])
    assert np.array_equal(frames[1], img[1])
    assert np.array_equal(frames[2], img[2])


def test_get_frames_from_array_time_not_first_axis():
    img = np.arange(60).reshape(4, 3, 5).astype(np.uint8)  # YTX
    frames = get_frames_from_array(img, "YTX")
    
    assert len(frames) == 3
    assert frames[0].shape == (4, 5)
    assert np.array_equal(frames[0], img[:, 0, :])
    assert np.array_equal(frames[1], img[:, 1, :])


def test_get_frames_from_array_complex_axis_order():
    img = np.arange(120).reshape(2, 3, 4, 5).astype(np.uint8)  # TCYX
    frames = get_frames_from_array(img, "TCYX")
    
    assert len(frames) == 2
    assert frames[0].shape == (3, 4, 5)
    assert np.array_equal(frames[0], img[0])
