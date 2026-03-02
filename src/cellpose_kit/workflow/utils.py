from typing import Any

from numpy.typing import NDArray


def get_axis(axis_order: str, axis: str) -> int | None:
    if axis not in axis_order:
        return None
    return axis_order.index(axis)


def validate_axis_order(img: NDArray[Any], axis_order: str) -> None:
    if len(axis_order) != img.ndim:
        raise ValueError(f"axis_order '{axis_order}' length ({len(axis_order)}) does not match image.ndim ({img.ndim}). Shape: {img.shape}")
    
    if len(set(axis_order)) != len(axis_order):
        raise ValueError(f"axis_order '{axis_order}' contains duplicate axis labels.")


def count_channels(img: NDArray[Any], axis_order: str) -> int:
    validate_axis_order(img, axis_order)
    channel_axis = get_axis(axis_order, "C")
    if channel_axis is None:
        return 1
    return int(img.shape[channel_axis])





