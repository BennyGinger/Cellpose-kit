from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING, TypeVar
from threading import Lock

from numpy.typing import NDArray
import numpy as np

from cellpose_kit.backend.runtime import ModelContext
from cellpose_kit.backend.factory import load_backend
from cellpose_kit.backend.utils import validate_image_channels

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cellpose.denoise import CellposeDenoiseModel

T = TypeVar('T', bound=np.generic)

logger = logging.getLogger('cellpose_kit')


def setup_cellpose(user_settings: dict[str, Any], threading: bool = False, use_nuclear_channel: bool = False, do_denoise: bool = False, model: CellposeModel | CellposeDenoiseModel | None = None) -> ModelContext:
    """
    Setup Cellpose model and evaluation parameters once for reuse.
    
    Policy:
        - For nuclear channel handling:
            - v3: sets channels=[1,2] with 2 being the nuclear channel so the input image should have at least 2 channels if use_nuclear_channel=True
            - v4: Informational only (expects 3-channel input, will ignore channel settings
    
    Parameters:
        user_settings (dict): Dictionary containing the settings for Cellpose given by the user.
        threading (bool): If True, adds a lock for thread-safe inference.
        use_nuclear_channel (bool): If True, configures for nuclear channel usage.
        do_denoise (bool): If True, applies denoising to the input images. Only valid in v3, will be ignored in v4.
        model (Any): Optional pre-initialized Cellpose model instance to use instead of creating a new one. Default is None.

    Returns:
        dict: Complete settings ready for run_cellpose, includes 'model' and 'eval_params'
    """
    backend, backend_name = load_backend()
    
    # If an existing model is provided, use it; otherwise, create a new one
    if model is not None:
        model_instance = model
    else:
        model_instance = backend.init_model(user_settings, do_denoise)
    eval_params = backend.configure_eval_params(user_settings, use_nuclear_channel, do_denoise)

    if model is not None:
        logger.info(f"Cellpose {backend_name} model reused from cache.")
    else:
        logger.info(f"Cellpose {backend_name} model initialized.")

    model_context = ModelContext(model=model_instance, 
                                 eval_params=eval_params,
                                 use_nuclear_channel=use_nuclear_channel,
                                 model_names=backend.model_names, 
                                 backend_name=backend_name)

    if threading:
        logger.info("Threading enabled: Adding lock for thread-safe model inference")
        model_context.lock = Lock()

    return model_context

def run_cellpose(img: NDArray[T] | list[NDArray[T]], axis_order: str, model_context: ModelContext) -> tuple[NDArray[T] | list[NDArray[T]], list[NDArray[T] | list[NDArray[T]]], NDArray[T] | list[NDArray[T]]]:
    """
    Run Cellpose segmentation using pre-configured settings.
    
    Policy:
        - v3: Flexible channel input, but must have >= 2 channels if nuclear mode enabled
        - v4: Flexible channel input, but must have 3 channels if nuclear mode enabled
    
    Parameters:
        img: Input image(s) - NDArray or list of NDArrays
        axis_order: String representing the axis order of the input image (e.g., "ZYX", "YXC", etc.).
        configured_settings: Settings from setup_cellpose(), must contain 'model' and 'eval_params'

    Returns:
        tuple: (masks, flows, styles)
        - masks: NDArray (for single/batch) or list[NDArray] (for list input)
        - flows: list[NDArray] (for single/batch) or list[list[NDArray]] (for list input)
        - styles: NDArray (for single/batch) or list[NDArray] (for list input)
    """
    model = model_context.model
    eval_params = model_context.eval_params
    
    # Validate image channels against configuration
    validate_image_channels(img, 
                            axis_order=axis_order, 
                            eval_params=eval_params,
                            backend_name=model_context.backend_name, 
                            use_nuclear_channel=model_context.use_nuclear_channel)
        
    lock = model_context.lock
    
    if lock is not None:
        with lock:
            logger.info("Threading lock acquired, running inference.")
            results = model.eval(img, **eval_params)
            return results[:3]  # masks, flows, styles, ignore the last returned value, if any.
    
    logger.info("No threading lock provided, running inference directly.")
    results = model.eval(img, **eval_params)
    return results[:3]  # masks, flows, styles, ignore the last returned value, if any.



