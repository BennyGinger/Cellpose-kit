"""Cellpose Kit - Clean API for Cellpose v3/v4"""
import logging
from typing import Any, Union, List
from threading import Lock

from numpy.typing import NDArray

from cellpose_kit.compat import get_cellpose_version
from cellpose_kit.utils import validate_image_channels

backend_name = get_cellpose_version()

match backend_name:
    case "v3":
        from cellpose_kit.backend.v3 import configure_eval_params, init_model
    case "v4":
        from cellpose_kit.backend.v4 import configure_eval_params, init_model
    case _:
        raise ImportError(f"Unsupported backend: {backend_name}")

logger = logging.getLogger('cellpose_kit')

def setup_cellpose(
    cellpose_settings: dict[str, Any],
    threading: bool = False,
    use_nuclear_channel: bool = False,
    do_denoise: bool = False,
    model: Any = None
) -> dict[str, Any]:
    """
    Setup Cellpose model and evaluation parameters once for reuse.
    
    Parameters:
        cellpose_settings (dict): Dictionary containing the settings for Cellpose.
        threading (bool): If True, adds a lock for thread-safe inference.
        use_nuclear_channel (bool): If True, configures for nuclear channel usage.
                                  - v3: Sets channels=[1,2] 
                                  - v4: Informational only (expects 3-channel input)
        do_denoise (bool): If True, applies denoising to the input images. Only valid in v3, will be ignored in v4.
        model (Any): Optional pre-initialized Cellpose model instance to use instead of creating a new one. Default is None.

    Returns:
        dict: Complete settings ready for run_cellpose, includes 'model' and 'eval_params'
    """
    # If an existing model is provided, use it; otherwise, create a new one
    if model is not None:
        model_instance = model
    else:
        model_instance = init_model(cellpose_settings, do_denoise)
    eval_params = configure_eval_params(cellpose_settings, use_nuclear_channel, do_denoise)

    if model is not None:
        logger.info(f"Cellpose {backend_name} model reused from cache.")
    else:
        logger.info(f"Cellpose {backend_name} model initialized.")

    configured_settings = {
        'model': model_instance,
        'eval_params': eval_params
    }

    if threading:
        logger.info("Threading enabled: Adding lock for thread-safe model inference")
        configured_settings['lock'] = Lock()

    return configured_settings

def run_cellpose(img: Union[NDArray, List[NDArray]], configured_settings: dict[str, Any]) -> tuple[Union[NDArray, List[NDArray]], list, Union[NDArray, List[NDArray]]]:
    """
    Run Cellpose segmentation using pre-configured settings.
    
    Parameters:
        img: Input image(s) - NDArray or list of NDArrays
             - v3: Flexible channel input, but must have >= 2 channels if nuclear mode enabled
             - v4: Must have 3 channels
        configured_settings: Settings from setup_cellpose(), must contain 'model' and 'eval_params'

    Returns:
        tuple: (masks, flows, styles)
        - masks: NDArray (for single/batch) or list[NDArray] (for list input)
        - flows: list[NDArray] (for single/batch) or list[list[NDArray]] (for list input)
        - styles: NDArray (for single/batch) or list[NDArray] (for list input)
        
    Raises:
        KeyError: If configured_settings is missing required keys
        TypeError: If configured_settings is not from setup_cellpose()
        ValueError: If image channels don't match nuclear channel requirements
    """
    try:
        model = configured_settings['model']
        eval_params = configured_settings['eval_params']
    except KeyError as e:
        raise KeyError(f"Invalid configured_settings: missing {e}. Use setup_cellpose() to create valid settings.") from e
    
    # Validate image channels against configuration
    validate_image_channels(img, eval_params, backend_name)
        
    lock = configured_settings.get('lock', None)
    
    if lock is not None:
        with lock:
            logger.info("Threading lock acquired, running inference.")
            return model.eval(img, **eval_params)
    
    logger.info("No threading lock provided, running inference directly.")
    return model.eval(img, **eval_params)



if __name__ == "__main__":
    from tifffile import imread
    import numpy as np
    import sys
    from pathlib import Path as TestPath
    
    
    folder_path = TestPath("/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Images_Registered")
    
    # Load z-stack
    # img = np.array([imread(p) for p in folder_path.rglob("*.tif") if 'f0001' in p.name and 'GFP' in p.name])
    # cellpose_settings = {
    #     "stitch_threshold": 0.75,
    #     # "do_3D": True
    # }
    
    # Load list
    img = [imread(p) for p in folder_path.rglob("*.tif") if 'f0001' in p.name and 'GFP' in p.name]
    cellpose_settings = {}
    
    # Load array
    # img = imread(TestPath("/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Images_Registered/GFP_s01_f0001_z0008.tif"))
    # cellpose_settings = {}
    
    # Load config
    settings = setup_cellpose(cellpose_settings, threading=False, use_nuclear_channel=False)
    
    # Run segmentation
    m,f,s = run_cellpose(img, settings)
    
    if isinstance(m, list):
        print(len(m))
        print(m[0].shape)
    else:
        print(m.shape)
    
    if isinstance(f[0], list):
        print(len(f))
        print(f[0][0].shape, f[0][1].shape, f[0][2].shape)
    else:
        print(f[0].shape, f[1].shape, f[2].shape)