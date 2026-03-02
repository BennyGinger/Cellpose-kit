from __future__ import annotations

import pytest
import numpy as np

from cellpose_kit.workflow.validation import validate_axis_order, validate_channel_requirements, ensure_list


def test_validate_axis_order_correct():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    validate_axis_order(img, "YXC")  # Should not raise


def test_validate_axis_order_length_mismatch():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        validate_axis_order(img, "YX")


def test_validate_axis_order_duplicate_labels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="duplicate axis labels"):
        validate_axis_order(img, "YYC")


def test_validate_channel_requirements_v3_nuclear_needs_2_channels():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 2 channels"):
        validate_channel_requirements(img, "YX", "v3", use_nuclear_channel=True)


def test_validate_channel_requirements_v3_nuclear_with_2_channels():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    validate_channel_requirements(img, "YXC", "v3", use_nuclear_channel=True)  # Should not raise


def test_validate_channel_requirements_v4_nuclear_needs_2_or_3():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="2 or 3 channels"):
        validate_channel_requirements(img, "YX", "v4", use_nuclear_channel=True)


def test_validate_channel_requirements_v4_nuclear_with_2_channels():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    validate_channel_requirements(img, "YXC", "v4", use_nuclear_channel=True)  # Should not raise (will be padded)


def test_validate_channel_requirements_v4_nuclear_with_3_channels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    validate_channel_requirements(img, "YXC", "v4", use_nuclear_channel=True)  # Should not raise


def test_validate_channel_requirements_invalid_backend():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported backend"):
        validate_channel_requirements(img, "YXC", "v5", use_nuclear_channel=False)


def test_ensure_list_with_list():
    result = ensure_list([1, 2, 3], 3, "test")
    assert result == [1, 2, 3]


def test_ensure_list_with_single_value():
    result = ensure_list("value", 1, "test")
    assert result == ["value"]


def test_ensure_list_wrong_length():
    with pytest.raises(ValueError, match="length 2 but expected 3"):
        ensure_list([1, 2], 3, "test")


def test_ensure_list_single_value_multi_frame():
    with pytest.raises(ValueError, match="non-list"):
        ensure_list("value", 5, "test")
