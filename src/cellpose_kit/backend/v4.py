from pathlib import Path
from typing import Any
import logging

from cellpose.models import MODEL_NAMES, normalize_default


logger = logging.getLogger('cellpose_kit_v4')

DEFAULT_MODEL = 'cpsam'

MOD_SETS = {
    "gpu": True, # Use GPU for processing, set to False for CPU
    "pretrained_model": DEFAULT_MODEL, # Full path to pretrained cellpose model(s), if None or False, no model loaded.
    "device": None, #Device used for model running / training (torch.device("cuda") or torch.device("cpu")), overrides gpu input, recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
    "use_bfloat16": True, # Use 16bit float precision instead of 32bit for model weights. Default to 16bit (True).
}

EVAL_SETS = {
    "batch_size": 8, # number of 256x256 patches to run simultaneously on the GPU (smaller or bigger depending on GPU memory). Defaults to 64.
    "resample": True, # run dynamics at original image size (will be slower but create more accurate boundaries).
    "z_axis": None, # z axis in element of list x, or of np.ndarray x. if None, z dimension is automatically determined. Defaults to None.
    "normalize": normalize_default, # if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; can also pass dictionary of parameters (all keys are optional, default values shown in normalize_default)
    "invert": False, # invert image pixel intensity before running network. Defaults to False.
    "diameter": None, # diameters are used to rescale the image to 30 pix cell diameter.
    "flow_threshold": 0.4, # flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
    "cellprob_threshold": 0.0, # all pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
    "do_3D": False, # set to True to run 3D segmentation on 3D/4D image input. Defaults to False.
    "anisotropy": None, # for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
    "flow3D_smooth": 0, # if do_3D and flow3D_smooth>0, smooth flows with gaussian filter of this stddev. Defaults to 0.
    "stitch_threshold": 0, # if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation. Defaults to 0.0.
    "min_size": 15, # all ROIs below this size, in pixels, will be discarded. Defaults to 15.
    "max_size_fraction": 0.4, # max_size_fraction (float, optional): Masks larger than max_size_fraction of total image size are removed. Default is 0.4.
    "niter": None, # Number of iterations for dynamics computation. if None, it is set proportional to the diameter. Defaults to None.
    "augment": False, # tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
    "tile_overlap": 0.1, # fraction of overlap of tiles when computing flows. Defaults to 0.1.
    "bsize": 256, # block size for tiles, recommended to keep at 224, like in training. Defaults to 224.
    "compute_masks": True, # Whether or not to compute dynamics and return masks. Returns empty array if False. Defaults to True.
    "progress": None, # pyqt progress bar. Defaults to None.
    }

def configure_model(cellpose_settings: dict[str, Any]) -> dict[str, Any]:
    """
    Configure the model settings based on user input. If missing or invalid, revert to defaults.
    For Cellpose v4, handles model name validation and file path checking.
    
    Returns:
        dict: Updated model settings dictionary
    """
    mod_sets = MOD_SETS.copy()
    
    # Update with user-provided settings
    overwrites = {k: v for k, v in cellpose_settings.items() if k in mod_sets}
    mod_sets.update(overwrites)

    # Handle deprecated 'model_type' parameter (v4 uses 'pretrained_model')
    if 'model_type' in cellpose_settings and cellpose_settings['model_type'] is not None:
        logger.warning("⚠️ 'model_type' is deprecated in Cellpose v4. Use 'pretrained_model' instead.")
        model_type = cellpose_settings['model_type']
        
        # Check if it's a valid model name or file path
        if model_type in MODEL_NAMES or Path(model_type).is_file():
            mod_sets['pretrained_model'] = model_type
        else:
            logger.warning(f"⚠️ Model type '{model_type}' is not valid, using default model '{DEFAULT_MODEL}'.")
            mod_sets['pretrained_model'] = DEFAULT_MODEL

    # Validate pretrained_model setting
    pretrained_model = mod_sets['pretrained_model']
    if pretrained_model:
        # For v4, accept model names (like 'cpsam') or valid file paths
        is_valid_model_name = pretrained_model in MODEL_NAMES
        is_valid_file_path = Path(pretrained_model).is_file()
        
        if not (is_valid_model_name or is_valid_file_path):
            logger.warning(f"⚠️ Pretrained model '{pretrained_model}' not found. Using default model '{DEFAULT_MODEL}'.")
            mod_sets['pretrained_model'] = DEFAULT_MODEL
    else:
        # If no model specified, use default
        mod_sets['pretrained_model'] = DEFAULT_MODEL
        
    return mod_sets

def configure_eval_params(cellpose_settings: dict[str, Any], use_nuclear_channel: bool = False) -> dict[str, Any]:
    """
    Configure the evaluation parameters based on user input. If missing or invalid, revert to defaults.
    
    Note: For Cellpose v4, nuclear channel handling is different from v3:
    - v3 uses 'channels=[1,2]' parameter to specify cytoplasm and nucleus channels
    - v4 expects 3-channel input where nuclear information is pre-incorporated
    
    Parameters:
        cellpose_settings: User-provided settings
        use_nuclear_channel: For v4, this parameter is informational only since
                           nuclear information should be pre-composed in the 3-channel input
    
    Returns:
        dict: Updated evaluation parameters dictionary
    """
    eval_params = EVAL_SETS.copy()

    # Update with user-provided settings
    overwrites = {k: v for k, v in cellpose_settings.items() if k in eval_params}
    eval_params.update(overwrites)

    # Warn about nuclear channel usage in v4
    if use_nuclear_channel:
        logger.info("ℹ️ Nuclear channel usage in Cellpose v4: Ensure your images have 3 channels with nuclear information pre-incorporated. The 'use_nuclear_channel' parameter doesn't modify v4 processing.")

    # Warn if user tries to use deprecated 'channels' parameter
    if 'channels' in cellpose_settings:
        logger.warning("⚠️ 'channels' parameter is deprecated in Cellpose v4. Ensure your input images have 3 channels with nuclear information already incorporated.")

    # Handle 3D segmentation configuration
    if eval_params["do_3D"]:
        eval_params['z_axis'] = 0
        eval_params['anisotropy'] = 2.0 if eval_params['anisotropy'] is None else eval_params['anisotropy']

    # Handle 3D stitching configuration
    if eval_params["stitch_threshold"] > 0.0:
        eval_params['do_3D'] = False
        
    return eval_params