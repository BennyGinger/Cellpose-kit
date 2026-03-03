from __future__ import annotations

import pytest
import numpy as np

from cellpose_kit.workflow.prep import prepare_streams
from cellpose_kit.workflow.models import InputStream


def test_prepare_streams_nuclear_mode_v3():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    streams, meta = prepare_streams(img, "YXC", "v3", use_nuclear_channel=True, do_3D=False)
    
    assert len(streams) == 1
    assert streams[0].stream_id == "stream0"
    assert streams[0].axis_order == "YXC"
    assert streams[0].meta["channel_index"] is None
    assert streams[0].meta["padded_to_3"] is False
    assert meta["use_nuclear_channel"] is True
    assert meta["split_channels"] is False


def test_prepare_streams_nuclear_mode_v4_with_2_channels():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    streams, meta = prepare_streams(img, "YXC", "v4", use_nuclear_channel=True, do_3D=False)
    
    assert len(streams) == 1
    assert streams[0].source_array.shape == (10, 10, 3)  # Padded to 3
    assert streams[0].meta["padded_to_3"] is True
    assert meta["any_padding_applied"] is True


def test_prepare_streams_nuclear_mode_v4_with_3_channels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    streams, meta = prepare_streams(img, "YXC", "v4", use_nuclear_channel=True, do_3D=False)
    
    assert len(streams) == 1
    assert streams[0].source_array.shape == (10, 10, 3)  # No padding needed
    assert streams[0].meta["padded_to_3"] is False
    assert meta["any_padding_applied"] is False


def test_prepare_streams_non_nuclear_split_channels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    streams, meta = prepare_streams(img, "YXC", "v3", use_nuclear_channel=False, do_3D=False)
    
    assert len(streams) == 3
    assert streams[0].stream_id == "ch0"
    assert streams[1].stream_id == "ch1"
    assert streams[2].stream_id == "ch2"
    assert streams[0].axis_order == "YX"  # Channel axis removed
    assert streams[0].meta["channel_index"] == 0
    assert meta["split_channels"] is True


def test_prepare_streams_non_nuclear_no_split_single_channel():
    img = np.zeros((10, 10), dtype=np.uint8)
    streams, meta = prepare_streams(img, "YX", "v3", use_nuclear_channel=False, do_3D=False)
    
    assert len(streams) == 1
    assert streams[0].stream_id == "stream0"
    assert streams[0].meta["channel_index"] is None
    assert meta["split_channels"] is False


def test_prepare_streams_invalid_backend():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported backend"):
        prepare_streams(img, "YXC", "v5", use_nuclear_channel=False, do_3D=False)


def test_prepare_streams_v3_nuclear_insufficient_channels():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 2 channels"):
        prepare_streams(img, "YX", "v3", use_nuclear_channel=True, do_3D=False)


def test_prepare_streams_metadata_complete():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    streams, meta = prepare_streams(img, "YXC", "v3", use_nuclear_channel=False, do_3D=False)
    
    assert meta["backend"] == "v3"
    assert meta["use_nuclear_channel"] is False
    assert meta["input_axis_order"] == "YXC"
    assert meta["input_shape"] == (10, 10, 2)
    assert meta["n_input_channels"] == 2
    assert "any_padding_applied" in meta
