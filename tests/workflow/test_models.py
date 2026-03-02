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


def test_segmentation_result_single():
    stream = StreamResult(
        stream_id="stream0",
        channel_index=None,
        masks=[np.zeros((10, 10))],
        flows=[[np.zeros((10, 10))]],
        styles=[np.zeros(64)],
        meta={}
    )
    
    seg_result = SegmentationResult(streams=[stream], meta={})
    single = seg_result.single()
    
    assert single.stream_id == "stream0"


def test_segmentation_result_single_raises_with_multiple_streams():
    streams = [
        StreamResult(stream_id="s1", channel_index=0, masks=[], flows=[], styles=[], meta={}),
        StreamResult(stream_id="s2", channel_index=1, masks=[], flows=[], styles=[], meta={}),
    ]
    
    seg_result = SegmentationResult(streams=streams, meta={})
    
    with pytest.raises(ValueError, match="Expected exactly 1 stream"):
        seg_result.single()


def test_segmentation_result_masks_by_channel():
    mask0 = [np.zeros((10, 10))]
    mask1 = [np.ones((10, 10))]
    
    streams = [
        StreamResult(stream_id="ch0", channel_index=0, masks=mask0, flows=[], styles=[], meta={}),
        StreamResult(stream_id="ch1", channel_index=1, masks=mask1, flows=[], styles=[], meta={}),
    ]
    
    seg_result = SegmentationResult(streams=streams, meta={})
    masks_dict = seg_result.masks_by_channel()
    
    assert 0 in masks_dict
    assert 1 in masks_dict
    assert masks_dict[0] == mask0
    assert masks_dict[1] == mask1


def test_segmentation_result_masks_by_channel_with_none():
    mask = [np.zeros((10, 10))]
    
    streams = [
        StreamResult(stream_id="stream0", channel_index=None, masks=mask, flows=[], styles=[], meta={}),
    ]
    
    seg_result = SegmentationResult(streams=streams, meta={})
    masks_dict = seg_result.masks_by_channel()
    
    assert 0 in masks_dict
    assert masks_dict[0] == mask
