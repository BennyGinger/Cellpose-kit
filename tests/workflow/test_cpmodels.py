from __future__ import annotations

import numpy as np
import pytest

from cellpose_kit.workflow.models import BatchResult, InferenceBatch, SegmentationResult


def test_inference_batch_creation() -> None:
    img = np.zeros((10, 10), dtype=np.uint8)
    batch = InferenceBatch(array=img, axes="YX", channel_index=0)

    assert batch.array is img
    assert batch.axes == "YX"
    assert batch.channel_index == 0


def test_masks_array_single_batch_without_time() -> None:
    masks = [np.full((3, 4), 5, dtype=np.uint8)]
    result = SegmentationResult(
        batches=[BatchResult(None, masks, [], [])],
        output_axes="YX",)

    np.testing.assert_array_equal(result.masks_array(), masks[0])


def test_masks_array_single_batch_with_time() -> None:
    masks = [np.full((3, 4), 1), np.full((3, 4), 2)]
    result = SegmentationResult(
        batches=[BatchResult(None, masks, [], [])],
        output_axes="TYX",)

    array = result.masks_array()
    assert array.shape == (2, 3, 4)
    np.testing.assert_array_equal(array[1], masks[1])


def test_masks_array_multiple_channels_and_time() -> None:
    channel_0 = [np.full((2, 3), 10), np.full((2, 3), 11)]
    channel_1 = [np.full((2, 3), 20), np.full((2, 3), 21)]
    result = SegmentationResult(
        batches=[BatchResult(0, channel_0, [], []),
                 BatchResult(1, channel_1, [], []),],
        output_axes="TCYX",)

    array = result.masks_array()
    assert array.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(array[1, 0], channel_0[1])
    np.testing.assert_array_equal(array[0, 1], channel_1[0])


def test_masks_array_rejects_multiple_batches_without_channel_axis() -> None:
    result = SegmentationResult(
        batches=[BatchResult(0, [np.zeros((3, 3))], [], []),
                 BatchResult(1, [np.ones((3, 3))], [], []),],
        output_axes="YX",)

    with pytest.raises(ValueError, match="one inference batch"):
        result.masks_array()


def test_masks_array_requires_channel_indices() -> None:
    result = SegmentationResult(
        batches=[BatchResult(None, [np.zeros((3, 3))], [], [])],
        output_axes="CYX",)

    with pytest.raises(ValueError, match="no channel index"):
        result.masks_array()
