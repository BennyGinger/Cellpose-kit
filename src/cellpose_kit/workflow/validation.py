from typing import Any

from numpy.typing import NDArray

from cellpose_kit.workflow.utils import count_channels, validate_axis_order


def validate_channel_requirements(img: NDArray[Any], axis_order: str, backend: str, use_nuclear_channel: bool) -> None:
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
    validate_axis_order(img, axis_order)
    
    if backend not in {"v3", "v4"}:
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'v3' or 'v4'. axis_order='{axis_order}', shape={img.shape}, use_nuclear_channel={use_nuclear_channel}")
    
    n_channels = count_channels(img, axis_order)
    
    if use_nuclear_channel:
        if backend == "v3":
            if n_channels < 2:
                raise ValueError(f"Cellpose backend={backend} with use_nuclear_channel=True requires at least 2 channels, but got n_channels={n_channels}. axis_order='{axis_order}', shape={img.shape}")
                                 
        elif backend == "v4":
            if n_channels < 2 or n_channels > 3:
                raise ValueError(f"Cellpose backend={backend} with use_nuclear_channel=True requires 2 or 3 channels, but got n_channels={n_channels}. axis_order='{axis_order}', shape={img.shape}. Note: 2 channels will be automatically padded to 3.")

def ensure_list(value: Any, expected_len: int, field_name: str) -> list[Any]:
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