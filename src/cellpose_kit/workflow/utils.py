from typing import Any

from numpy.typing import NDArray


def get_axis(axis_order: str, axis: str) -> int | None:
    if axis not in axis_order:
        return None
    return axis_order.index(axis)

def count_axis(img: NDArray[Any], axis_order: str, axis: str) -> int:
    """
    Count the number of elements along the specified axis in the image, based on the axis order.
    """
    n_axis = get_axis(axis_order, axis)
    if n_axis is None:
        return 1
    return int(img.shape[n_axis])





