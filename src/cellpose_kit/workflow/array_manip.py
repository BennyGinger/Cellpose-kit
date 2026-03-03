from typing import TypeVar

from numpy.typing import NDArray
import numpy as np

from cellpose_kit.workflow.utils import get_axis


T = TypeVar('T', bound=np.generic)

def pad_to_3_channels(img: NDArray[T], axis_order: str) -> NDArray[T]:
    """
    Pad a 2-channel image to 3 channels by adding a zero channel. The new channel will be added at the end of the channel axis.
    """
    
    channel_axis = get_axis(axis_order, "C")
    if channel_axis is None:
        raise ValueError(f"Cannot pad to 3 channels because axis_order '{axis_order}' has no 'C' axis. Shape: {img.shape}")
    
    n_channels = img.shape[channel_axis]
    if n_channels != 2:
        raise ValueError(f"pad_to_3_channels requires exactly 2 input channels but got {n_channels}. axis_order='{axis_order}', shape={img.shape}")

    pad_shape = list(img.shape)
    pad_shape[channel_axis] = 1
    zeros = np.zeros(pad_shape, dtype=img.dtype)
    return np.concatenate([img, zeros], axis=channel_axis)


def split_channels(img: NDArray[T], axis_order: str) -> tuple[list[NDArray[T]], str]:
    """
    Split a multi-channel image into separate arrays for each channel. The channel axis is removed from the output
    """
    
    channel_axis = get_axis(axis_order, "C")
    if channel_axis is None:
        raise ValueError(f"Cannot split channels because axis_order '{axis_order}' has no 'C' axis. Shape: {img.shape}")

    n_channels = img.shape[channel_axis]
    split_arrays = [np.take(img, indices=i, axis=channel_axis) for i in range(n_channels)]
    split_axis_order = axis_order.replace("C", "", 1)
    return split_arrays, split_axis_order

def get_frames_from_array(img: NDArray[T], axis_order: str) -> list[NDArray[T]]:
    """
    Extract frames from array along T axis, or return single-element list if no T axis.
    """
    frame_axis = get_axis(axis_order, "T")
    
    if frame_axis is None:
        return [img]

    n_frames = img.shape[frame_axis]
    frames = []
    for i in range(n_frames):
        slicer: list[slice | int] = [slice(None)] * img.ndim
        slicer[frame_axis] = i
        frames.append(img[tuple(slicer)])
    return frames