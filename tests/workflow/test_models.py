from __future__ import annotations

import pytest
import numpy as np

from cellpose_kit.workflow.models import InputStream, StreamResult, SegmentationResult


def test_input_stream_creation():
    img = np.zeros((10, 10), dtype=np.uint8)
    meta = {"channel_index": 0}
    
    stream = InputStream(
        source_array=img,
        axis_order="YX",
        stream_id="test",
        meta=meta
    )
    
    assert stream.stream_id == "test"
    assert stream.axis_order == "YX"
    assert np.array_equal(stream.source_array, img)
    assert stream.meta == {"channel_index": 0}


def test_stream_result_creation():
    masks = [np.zeros((10, 10), dtype=np.uint8)]
    flows = [[np.zeros((10, 10), dtype=np.float32)]]
    styles = [np.zeros(64, dtype=np.float32)]
    
    result = StreamResult(
        stream_id="ch0",
        channel_index=0,
        masks=masks,
        flows=flows,
        styles=styles,
        meta={"frames_count": 1}
    )
    
    assert result.stream_id == "ch0"
    assert result.channel_index == 0
    assert result.masks == masks
    assert result.meta["frames_count"] == 1


def test_segmentation_result_output_axis_order_property():
    seg_result = SegmentationResult(streams=[], meta={"output_axis_order": "TCYX"})
    assert seg_result.output_axis_order == "TCYX"


def test_segmentation_result_output_axis_order_none():
    seg_result = SegmentationResult(streams=[], meta={})
    assert seg_result.output_axis_order is None


def test_masks_array_single_stream_no_time():
    masks = [np.full((3, 4), 5, dtype=np.uint8)]
    stream = StreamResult(stream_id="stream0", channel_index=None, masks=masks, flows=[], styles=[], meta={})
    
    seg_result = SegmentationResult(streams=[stream], meta={"output_axis_order": "YX"})
    arr = seg_result.masks_array()
    
    assert arr.shape == (3, 4)
    assert np.array_equal(arr, masks[0])


def test_masks_array_single_stream_with_time():
    masks = [np.full((3, 4), 1, dtype=np.uint8), np.full((3, 4), 2, dtype=np.uint8)]
    stream = StreamResult(stream_id="stream0", channel_index=None, masks=masks, flows=[], styles=[], meta={})
    
    seg_result = SegmentationResult(streams=[stream], meta={"output_axis_order": "TYX"})
    arr = seg_result.masks_array()
    
    assert arr.shape == (2, 3, 4)
    assert np.array_equal(arr[0], masks[0])
    assert np.array_equal(arr[1], masks[1])


def test_masks_array_multi_stream_with_channel():
    ch0_masks = [np.full((2, 3), 10, dtype=np.uint8)]
    ch1_masks = [np.full((2, 3), 20, dtype=np.uint8)]
    
    streams = [
        StreamResult(stream_id="ch0", channel_index=0, masks=ch0_masks, flows=[], styles=[], meta={}),
        StreamResult(stream_id="ch1", channel_index=1, masks=ch1_masks, flows=[], styles=[], meta={}),
    ]
    
    seg_result = SegmentationResult(streams=streams, meta={"output_axis_order": "CYX"})
    arr = seg_result.masks_array()
    
    assert arr.shape == (2, 2, 3)
    assert np.array_equal(arr[0], ch0_masks[0])
    assert np.array_equal(arr[1], ch1_masks[0])


def test_masks_array_multi_stream_with_time_and_channel():
    ch0_masks = [np.full((2, 3), 10, dtype=np.uint8), np.full((2, 3), 11, dtype=np.uint8)]
    ch1_masks = [np.full((2, 3), 20, dtype=np.uint8), np.full((2, 3), 21, dtype=np.uint8)]
    
    streams = [
        StreamResult(stream_id="ch0", channel_index=0, masks=ch0_masks, flows=[], styles=[], meta={}),
        StreamResult(stream_id="ch1", channel_index=1, masks=ch1_masks, flows=[], styles=[], meta={}),
    ]
    
    seg_result = SegmentationResult(streams=streams, meta={"output_axis_order": "TCYX"})
    arr = seg_result.masks_array()
    
    assert arr.shape == (2, 2, 2, 3)
    assert np.array_equal(arr[0, 0], ch0_masks[0])
    assert np.array_equal(arr[1, 0], ch0_masks[1])
    assert np.array_equal(arr[0, 1], ch1_masks[0])
    assert np.array_equal(arr[1, 1], ch1_masks[1])


def test_masks_array_no_output_axis_order():
    stream = StreamResult(stream_id="stream0", channel_index=None, masks=[np.zeros((3, 3))], flows=[], styles=[], meta={})
    seg_result = SegmentationResult(streams=[stream], meta={})
    
    with pytest.raises(ValueError, match="no 'output_axis_order'"):
        seg_result.masks_array()


def test_masks_array_multiple_frames_without_t_axis():
    masks = [np.zeros((3, 3)), np.ones((3, 3))]
    stream = StreamResult(stream_id="stream0", channel_index=None, masks=masks, flows=[], styles=[], meta={})
    seg_result = SegmentationResult(streams=[stream], meta={"output_axis_order": "YX"})
    
    with pytest.raises(ValueError, match="Expected 1 mask"):
        seg_result.masks_array()


def test_masks_array_multiple_streams_without_c_axis():
    streams = [
        StreamResult(stream_id="ch0", channel_index=0, masks=[np.zeros((3, 3))], flows=[], styles=[], meta={}),
        StreamResult(stream_id="ch1", channel_index=1, masks=[np.ones((3, 3))], flows=[], styles=[], meta={}),
    ]
    seg_result = SegmentationResult(streams=streams, meta={"output_axis_order": "TYX"})
    
    with pytest.raises(ValueError, match="Expected 1 stream"):
        seg_result.masks_array()


def test_masks_array_channel_index_none_with_c_axis():
    streams = [
        StreamResult(stream_id="ch0", channel_index=None, masks=[np.zeros((3, 3))], flows=[], styles=[], meta={}),
    ]
    seg_result = SegmentationResult(streams=streams, meta={"output_axis_order": "CYX"})
    
    with pytest.raises(ValueError, match="channel_index=None"):
        seg_result.masks_array()
