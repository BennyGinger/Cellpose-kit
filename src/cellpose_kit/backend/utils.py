import logging
from typing import Any

from numpy.typing import NDArray


logger = logging.getLogger('cellpose_kit.utils')


def validate_image_channels(img: NDArray[Any] | list[NDArray[Any]], axis_order: str, eval_params: dict[str, Any], backend_name: str | None, use_nuclear_channel: bool = False) -> None:
    """
    Validate that input images have sufficient channels for the requested configuration.
    
    This function ensures that images have the required number of channels for Cellpose when using nuclear channel mode:
    - v4 + use_nuclear_channel=True: requires exactly 3 channels
    - v3 + use_nuclear_channel=True: requires at least 2 channels
    - use_nuclear_channel=False: no channel requirement
    
    Parameters:
        img: Input image(s) - NDArray or list of NDArrays
        axis_order: String representing the axis order of the input image (e.g., "ZYX", "YXC", etc.). The position of 'C' indicates which axis contains channels.
        eval_params: Evaluation parameters containing channel configuration
        backend_name: Cellpose backend version ("v3" or "v4")
        use_nuclear_channel: Whether nuclear channel mode is enabled
        
    Raises:
        ValueError: If image doesn't have sufficient channels for the configuration
    """
    
    def _get_channel_axis(axis_order: str) -> int | None:
        """Get the axis index of the channel dimension."""
        if 'C' not in axis_order:
            return None
        return axis_order.index('C')
    
    def _check_single_image(image: NDArray, channel_axis: int | None) -> None:
        """Check if a single image has the required number of channels."""
        
        if len(axis_order) != image.ndim:
            raise ValueError(
                f"axis_order '{axis_order}' length ({len(axis_order)}) does not match "
                f"image.ndim ({image.ndim}). Shape: {image.shape}"
            )
        
        if channel_axis is None:
            # No channel dimension specified - image is grayscale
            n_channels = 1
        else:
            # Get channel count from the specified axis
            if channel_axis >= image.ndim:
                raise ValueError(
                    f"Channel axis '{channel_axis}' is out of bounds for image with shape {image.shape}. "
                    f"Image has {image.ndim} dimensions but axis_order specifies channel at position {channel_axis}."
                )
            n_channels = image.shape[channel_axis]
        
        if use_nuclear_channel:
            if backend_name == "v4":
                # v4 with nuclear mode requires exactly 3 channels
                if n_channels != 3:
                    raise ValueError(
                        f"Cellpose v4 with nuclear channel mode requires exactly 3 channels, but got {n_channels}. "
                        f"Image shape: {image.shape}, axis_order: '{axis_order}'. "
                        "Please provide a 3-channel image (e.g., RGB or similar)."
                    )
            elif backend_name == "v3":
                # v3 with nuclear mode requires at least 2 channels
                if n_channels < 2:
                    raise ValueError(
                        f"Cellpose v3 with nuclear channel mode requires at least 2 channels, but got {n_channels}. "
                        f"Image shape: {image.shape}, axis_order: '{axis_order}'. "
                        "Please provide multi-channel image or set use_nuclear_channel=False."
                    )
    
    if backend_name is None:
        raise ValueError(f"Backend_name must be 'v3' or 'v4' but not {backend_name}")  
    
    # Validate channel axis
    channel_axis = _get_channel_axis(axis_order)
    
    # Validate based on input type
    if isinstance(img, list):
        for i, image in enumerate(img):
            try:
                _check_single_image(image, channel_axis)
            except ValueError as e:
                raise ValueError(f"Image {i} in list: {e}") from e
    else:
        _check_single_image(img, channel_axis)
        
    logger.debug(f"Image channel validation passed for {backend_name} (axis_order='{axis_order}')")
