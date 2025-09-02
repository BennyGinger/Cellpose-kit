"""Utility functions for Cellpose Kit"""
import logging
from typing import Any, Union, List
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger('cellpose_kit.utils')


def validate_image_channels(img: Union[NDArray, List[NDArray]], eval_params: dict[str, Any], backend_name: str) -> None:
    """
    Validate that input images have sufficient channels for the requested configuration.
    
    This function ensures that when nuclear channel mode is enabled in Cellpose v3,
    the input images have at least 2 channels to prevent runtime errors.
    
    Parameters:
        img: Input image(s) - NDArray or list of NDArrays
        eval_params: Evaluation parameters containing channel configuration
        backend_name: Cellpose backend version ("v3" or "v4")
        
    Raises:
        ValueError: If image doesn't have sufficient channels for the configuration
        
    Examples:
        >>> import numpy as np
        >>> img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)  # 2D grayscale
        >>> eval_params = {'channels': [1, 2]}  # Nuclear channel mode
        >>> validate_image_channels(img, eval_params, "v3")  # Raises ValueError
        
        >>> img = np.random.randint(0, 255, (256, 256, 2), dtype=np.uint8)  # Multi-channel
        >>> validate_image_channels(img, eval_params, "v3")  # Passes validation
    """
    # Only validate for v3 with nuclear channels configured
    if backend_name != "v3" or eval_params.get('channels') != [1, 2]:
        return
        
    def _check_single_image(image: NDArray) -> None:
        """Check if a single image has sufficient channels for nuclear mode."""
        if image.ndim == 2:
            # 2D grayscale image (H, W) - no channels
            raise ValueError(
                "Nuclear channel mode requires at least 2 channels, but got 2D grayscale image (H, W). "
                "Please provide multi-channel image with shape (H, W, C) where C >= 2, "
                "or set use_nuclear_channel=False for single-channel processing."
            )
        elif image.ndim == 3:
            # Could be (H, W, C) or (Z, H, W) 
            # Check if it has enough channels in the last dimension
            if image.shape[-1] < 2:
                raise ValueError(
                    f"Nuclear channel mode requires at least 2 channels, but got image with shape {image.shape}. "
                    "Channel dimension (last dimension) must be >= 2. "
                    "Please provide multi-channel image or set use_nuclear_channel=False."
                )
        elif image.ndim == 4:
            # Could be (Z, H, W, C) for 3D with channels
            if image.shape[-1] < 2:
                raise ValueError(
                    f"Nuclear channel mode requires at least 2 channels, but got 4D image with shape {image.shape}. "
                    "Channel dimension (last dimension) must be >= 2. "
                    "Please provide multi-channel image or set use_nuclear_channel=False."
                )
        # For higher dimensions, assume user knows what they're doing
    
    # Validate based on input type
    if isinstance(img, list):
        for i, image in enumerate(img):
            try:
                _check_single_image(image)
            except ValueError as e:
                raise ValueError(f"Image {i} in list: {e}") from e
    else:
        _check_single_image(img)
        
    logger.debug(f"Image channel validation passed for {backend_name} with nuclear channels")
