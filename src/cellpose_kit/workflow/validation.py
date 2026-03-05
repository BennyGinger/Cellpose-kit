from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray

from cellpose_kit.workflow.utils import count_axis


def _validate_axis_order(img: NDArray[Any], axis_order: str) -> None:
    if len(axis_order) != img.ndim:
        raise ValueError(f"axis_order '{axis_order}' length ({len(axis_order)}) does not match image.ndim ({img.ndim}). Shape: {img.shape}")
    
    if len(set(axis_order)) != len(axis_order):
        raise ValueError(f"axis_order '{axis_order}' contains duplicate axis labels.")

def _validate_channel_requirements(img: NDArray[Any], axis_order: str, backend: str, use_nuclear_channel: bool) -> None:
    """
    Validate that the input image meets channel requirements for the given backend and mode.
    
    Parameters:
        img: Input image array
        axis_order: String representing the axis order (e.g., "TCZYX", "YXC")
        backend: Cellpose backend ("v3" or "v4")
        use_nuclear_channel: Whether nuclear channel mode is enabled
        
    Raises:
        ValueError: If image doesn't meet channel requirements for the configuration
    """
    if backend not in {"v3", "v4"}:
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'v3' or 'v4'")
    
    n_channels = count_axis(img, axis_order, "C")
    
    if use_nuclear_channel:
        if backend == "v3":
            if n_channels < 2:
                raise ValueError(f"Cellpose backend={backend} with use_nuclear_channel=True requires at least 2 channels, but got n_channels={n_channels}. axis_order='{axis_order}', shape={img.shape}")
                                 
        elif backend == "v4":
            if n_channels < 2 or n_channels > 3:
                raise ValueError(f"Cellpose backend={backend} with use_nuclear_channel=True requires 2 or 3 channels, but got n_channels={n_channels}. axis_order='{axis_order}', shape={img.shape}. Note: 2 channels will be automatically padded to 3.")

def _validate_z_axis_requirements(img: NDArray[Any], axis_order: str, do_3D: bool) -> None:
    """
    Validate that the input image meets Z-axis requirements for 3D segmentation.
    
    Parameters:
        img: Input image array
        axis_order: String representing the axis order (e.g., "TCZYX", "YXC")
        do_3D: Whether 3D mode is enabled
    """
    n_z = count_axis(img, axis_order, "Z")
    
    if do_3D:
        if n_z < 2:
            raise ValueError(f"3D segmentation requires at least 2 Z-slices, but got n_z={n_z}. axis_order='{axis_order}', shape={img.shape}")

def validate_array(img: NDArray[Any], axis_order: str, backend: str, use_nuclear_channel: bool, do_3D: bool) -> None:
    """
    Validate that the input image meets all requirements for Cellpose inference based on the configuration.
    
    Parameters:
        img: Input image array
        axis_order: String representing the axis order (e.g., "TCZYX", "YXC")
        backend: Cellpose backend ("v3" or "v4")
        use_nuclear_channel: Whether nuclear channel mode is enabled
        do_3D: Whether 3D mode is enabled
    """
    _validate_axis_order(img, axis_order)
    _validate_channel_requirements(img, axis_order, backend, use_nuclear_channel)
    _validate_z_axis_requirements(img, axis_order, do_3D)

def _ensure_list(value: Any, expected_len: int, field_name: str) -> list[Any]:
    """
    Ensure that the value is a list of the expected length. If it's not a list but expected_len is 1, wrap it in a list.
    """
    
    if isinstance(value, list):
        if len(value) != expected_len:
            raise ValueError(f"Cellpose returned '{field_name}' length {len(value)} but expected {expected_len} frames.")
        return value
    
    if expected_len == 1:
        return [value]
    
    raise ValueError(f"Cellpose returned non-list '{field_name}' for multi-frame input ({expected_len} frames).")


def ensure_lists(values: Sequence[Any], expected_len: int, field_names: Sequence[str]) -> tuple[list[Any], ...]:
    """
    Ensure that multiple values are lists of the expected length. Processes all items in one call.
    
    Parameters:
        values: Tuple of values to process
        expected_len: Expected length for each value
        field_names: Tuple of field names corresponding to each value (for error messages)
    
    Returns:
        Tuple of processed lists, one for each input value
    """
    if len(values) != len(field_names):
        raise ValueError(f"Number of values ({len(values)}) does not match number of field names ({len(field_names)})")
    
    return tuple(_ensure_list(value, expected_len, field_name) for value, field_name in zip(values, field_names))