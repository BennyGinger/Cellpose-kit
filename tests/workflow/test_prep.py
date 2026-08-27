from __future__ import annotations

import numpy as np
import pytest

from cellpose_kit.workflow.prep import prepare_batches


def test_prepare_batches_nuclear_mode_v3() -> None:
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    prepared = prepare_batches(img, "YXC", "v3", True, False)

    assert len(prepared.batches) == 1
    assert prepared.batches[0].axes == "YXC"
    assert prepared.batches[0].channel_index is None
    assert prepared.output_axes == "YX"


def test_prepare_batches_nuclear_mode_v4_pads_two_channels() -> None:
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    prepared = prepare_batches(img, "YXC", "v4", True, False)

    assert prepared.batches[0].array.shape == (10, 10, 3)
    assert prepared.output_axes == "YX"


def test_prepare_batches_nuclear_mode_v4_keeps_three_channels() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    prepared = prepare_batches(img, "YXC", "v4", True, False)

    assert prepared.batches[0].array.shape == (10, 10, 3)


def test_prepare_batches_processes_channels_independently() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    prepared = prepare_batches(img, "YXC", "v3", False, False)

    assert len(prepared.batches) == 3
    assert [batch.channel_index for batch in prepared.batches] == [0, 1, 2]
    assert all(batch.axes == "YX" for batch in prepared.batches)
    assert prepared.output_axes == "YXC"


def test_prepare_batches_without_channel_axis() -> None:
    img = np.zeros((10, 10), dtype=np.uint8)
    prepared = prepare_batches(img, "YX", "v3", False, False)

    assert len(prepared.batches) == 1
    assert prepared.batches[0].array is img
    assert prepared.batches[0].axes == "YX"
    assert prepared.output_axes == "YX"


def test_prepare_batches_removes_singleton_channel_axis() -> None:
    img = np.zeros((1, 10, 10), dtype=np.uint8)
    prepared = prepare_batches(img, "CYX", "v3", False, False)

    assert prepared.batches[0].array.shape == (10, 10)
    assert prepared.batches[0].axes == "YX"
    assert prepared.output_axes == "YX"


def test_prepare_batches_rejects_invalid_backend() -> None:
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported backend"):
        prepare_batches(img, "YXC", "v5", False, False)


def test_prepare_batches_rejects_missing_nuclear_channel() -> None:
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 2 channels"):
        prepare_batches(img, "YX", "v3", True, False)
