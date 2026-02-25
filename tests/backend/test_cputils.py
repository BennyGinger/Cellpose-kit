import numpy as np
import pytest

from cellpose_kit.backend.utils import validate_image_channels


def test_v4_requires_three_channels_last_axis() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    validate_image_channels(img, "YXC", {"channels": [1, 2]}, "v4")


def test_v4_raises_with_two_channels() -> None:
    img = np.zeros((64, 64, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires exactly 3 channels"):
        validate_image_channels(img, "YXC", {"channels": [1, 2]}, "v4")


def test_v4_uses_channel_axis_from_order() -> None:
    img = np.zeros((3, 64, 64), dtype=np.uint8)
    validate_image_channels(img, "CYX", {"channels": [1, 2]}, "v4")


def test_v4_raises_when_axis_order_has_no_channel() -> None:
    img = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires exactly 3 channels"):
        validate_image_channels(img, "YX", {"channels": [1, 2]}, "v4")


def test_axis_order_length_mismatch() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="axis_order .* length .* does not match"):
        validate_image_channels(img, "YX", {"channels": [1, 2]}, "v4")


def test_v3_nuclear_mode_requires_two_channels() -> None:
    img = np.zeros((64, 64, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires at least 2 channels"):
        validate_image_channels(img, "YXC", {"channels": [1, 2]}, "v3")


def test_v3_non_nuclear_mode_allows_single_channel() -> None:
    img = np.zeros((64, 64, 1), dtype=np.uint8)
    validate_image_channels(img, "YXC", {"channels": [0, 0]}, "v3")


def test_list_input_reports_index() -> None:
    imgs = [np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 2), dtype=np.uint8)]
    with pytest.raises(ValueError, match="Image 1 in list"):
        validate_image_channels(imgs, "YXC", {"channels": [1, 2]}, "v4")


def test_backend_name_required() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Backend_name must be"):
        validate_image_channels(img, "YXC", {"channels": [1, 2]}, None)
