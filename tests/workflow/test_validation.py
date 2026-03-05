from __future__ import annotations

import pytest
import numpy as np

from cellpose_kit.workflow.validation import _validate_axis_order, _validate_channel_requirements, _validate_z_axis_requirements, _ensure_list, ensure_lists


def test_validate_axis_order_correct():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _validate_axis_order(img, "YXC")  # Should not raise


def test_validate_axis_order_length_mismatch():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        _validate_axis_order(img, "YX")


def test_validate_axis_order_duplicate_labels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="duplicate axis labels"):
        _validate_axis_order(img, "YYC")


def test_validate_channel_requirements_v3_nuclear_needs_2_channels():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 2 channels"):
        _validate_channel_requirements(img, "YX", "v3", use_nuclear_channel=True)


def test_validate_channel_requirements_v3_nuclear_with_2_channels():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    _validate_channel_requirements(img, "YXC", "v3", use_nuclear_channel=True)  # Should not raise


def test_validate_channel_requirements_v4_nuclear_needs_2_or_3():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="2 or 3 channels"):
        _validate_channel_requirements(img, "YX", "v4", use_nuclear_channel=True)


def test_validate_channel_requirements_v4_nuclear_with_2_channels():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    _validate_channel_requirements(img, "YXC", "v4", use_nuclear_channel=True)  # Should not raise (will be padded)


def test_validate_channel_requirements_v4_nuclear_with_3_channels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _validate_channel_requirements(img, "YXC", "v4", use_nuclear_channel=True)  # Should not raise


def test_validate_channel_requirements_invalid_backend():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported backend"):
        _validate_channel_requirements(img, "YXC", "v5", use_nuclear_channel=False)


def test_validate_z_axis_requirements_3d_disabled_no_z_axis_ok():
    img = np.zeros((10, 10), dtype=np.uint8)
    _validate_z_axis_requirements(img, "YX", do_3D=False)


def test_validate_z_axis_requirements_3d_enabled_needs_at_least_2_z():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 2 Z-slices"):
        _validate_z_axis_requirements(img, "YX", do_3D=True)


def test_validate_z_axis_requirements_3d_enabled_with_valid_z_axis():
    img = np.zeros((3, 10, 10), dtype=np.uint8)
    _validate_z_axis_requirements(img, "ZYX", do_3D=True)


def test_ensure_list_with_list():
    result = _ensure_list([1, 2, 3], 3, "test")
    assert result == [1, 2, 3]


def test_ensure_list_with_single_value():
    result = _ensure_list("value", 1, "test")
    assert result == ["value"]


def test_ensure_list_wrong_length():
    with pytest.raises(ValueError, match="length 2 but expected 3"):
        _ensure_list([1, 2], 3, "test")


def test_ensure_list_single_value_multi_frame():
    with pytest.raises(ValueError, match="non-list"):
        _ensure_list("value", 5, "test")


def test_ensure_list_multiple_basic():
    masks, flows, styles = ensure_lists(
        ("mask_value", "flow_value", "style_value"),
        1,
        ("masks", "flows", "styles")
    )
    assert masks == ["mask_value"]
    assert flows == ["flow_value"]
    assert styles == ["style_value"]


def test_ensure_list_multiple_with_lists():
    masks, flows, styles = ensure_lists(
        ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
        3,
        ("masks", "flows", "styles")
    )
    assert masks == [1, 2, 3]
    assert flows == [4, 5, 6]
    assert styles == [7, 8, 9]


def test_ensure_list_multiple_mismatched_field_names():
    with pytest.raises(ValueError, match="Number of values"):
        ensure_lists(
            ("a", "b", "c"),
            1,
            ("field1", "field2")  # Only 2 names for 3 values
        )


def test_ensure_list_multiple_wrong_length():
    with pytest.raises(ValueError, match="length 2 but expected 3"):
        ensure_lists(
            ([1, 2], [3, 4], [5, 6]),
            3,
            ("masks", "flows", "styles")
        )


def test_ensure_list_multiple_mixed_types():
    # Test with scalar value and list containing 2 items, all with expected_len=2
    masks, flows = ensure_lists(
        ([1, 2], [3, 4]),
        2,
        ("masks", "flows")
    )
    assert masks == [1, 2]
    assert flows == [3, 4]
