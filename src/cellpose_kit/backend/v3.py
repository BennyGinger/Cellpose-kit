from pathlib import Path
from typing import Any
import logging

from cellpose.models import MODEL_NAMES, normalize_default


logger = logging.getLogger('cellpose_kit_v3')

DEFAULT_MODEL = "cyto2"

MOD_SETS = {
    "gpu": True, # Use GPU for processing, set to False for CPU
    "model_type": DEFAULT_MODEL, # Model type to use
    "pretrained_model": False, # Path to pretrained cellpose model.
    "diam_mean": 30., # Mean "diameter", 30. is built-in value for "cyto" model; 17. is built-in value for "nuclei" model; if saved in custom model file (cellpose>=2.0) then it will be loaded automatically and overwrite this value.
    "device": None, #Device used for model running / training (torch.device("cuda") or torch.device("cpu")), overrides gpu input, recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
    "nchan": 2, # Number of channels to use as input to network, default is 2 (cyto + nuclei) or (nuclei + zeros).
    "mkldnn": True, # Use MKLDNN for CPU inference, faster but not always supported.
    "pretrained_model_ortho": None, # Path or model_name for pretrained cellpose model for ortho views in 3D.
    "backbone": "default", # Type of network ("default" is the standard res-unet, "transformer" for the segformer).
}

EVAL_SETS = {
    "batch_size": 8, # number of 256x256 patches to run simultaneously on the GPU (smaller or bigger depending on GPU memory). Defaults to 64.
    "resample": True, # run dynamics at original image size (will be slower but create more accurate boundaries).
    "channels": None, # list of channels, either of length 2 or of length number of images by 2. First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue). Defaults to None.
    "channel_axis": None, # channel axis in element of list x, or of np.ndarray x.
    "z_axis": None, # z axis in element of list x, or of np.ndarray x. if None, z dimension is automatically determined. Defaults to None.
    "normalize": normalize_default, # if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; can also pass dictionary of parameters (all keys are optional, default values shown in normalize_default)
    "invert": False, # invert image pixel intensity before running network. Defaults to False.
    "rescale": None, # resize factor for each image, if None, set to 1.0; (only used if diameter is None). Defaults to None.
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
    "interp": True, # interpolate during 2D dynamics (not available in 3D) . Defaults to True.
    "compute_masks": True, # Whether or not to compute dynamics and return masks. Returns empty array if False. Defaults to True.
    "progress": None, # pyqt progress bar. Defaults to None.
    }


def configure_model(cellpose_settings: dict[str, Any]) -> dict[str, Any]:
    """
    Configure the model settings based on user input. If missing or invalid, revert to defaults. It uses the default values from cellpose.
    Returns the updated model settings as a dictionary.
    """
    
    mod_sets = MOD_SETS.copy()
    
    overwrites = {k: v for k, v in cellpose_settings.items() if k in mod_sets}
    mod_sets.update(overwrites)

    # Check if pretrained model is provided and is a valid file path
    if mod_sets['pretrained_model']:
        valid_path = Path(mod_sets['pretrained_model']).is_file()
        if valid_path:
            # If a valid custom model file is provided, ignore model_type
            mod_sets['model_type'] = None
        else:
            logger.warning(f"⚠️ Pretrained model not found: {mod_sets['pretrained_model']}, reverting to default model.")
            mod_sets['model_type'] = DEFAULT_MODEL
            mod_sets['pretrained_model'] = None

    # Check that model_type is not a 'pretrained_model' instead
    if mod_sets['model_type'] is not None:
        if Path(mod_sets['model_type']).is_file():
            mod_sets['pretrained_model'] = mod_sets['model_type']
            mod_sets['model_type'] = None
    
    # Check if model_type is valid
    if mod_sets['model_type'] is not None:
        if mod_sets['model_type'] not in MODEL_NAMES:
            logger.warning(f"⚠️ Unknown model type: {mod_sets['model_type']}, available models are: {MODEL_NAMES}, reverting to default model.")
            mod_sets['model_type'] = DEFAULT_MODEL

    return mod_sets

def configure_eval_params(cellpose_settings: dict[str, Any], use_nuclear_channel: bool = False) -> dict[str, Any]:
    """
    Configure the evaluation parameters based on user input. If missing or invalid, revert to defaults.
    
    Returns the updated evaluation parameters as a dictionary.
    """
    eval_params = EVAL_SETS.copy()

    overwrites = {k: v for k, v in cellpose_settings.items() if k in eval_params}
    eval_params.update(overwrites)

    if use_nuclear_channel:
        eval_params['channels'] = [1,2]

    # If user wants to do 3D segmentation
    if eval_params["do_3D"]:
        eval_params['z_axis'] = 0
        eval_params['anisotropy'] = 2.0 if eval_params['anisotropy'] is None else eval_params['anisotropy']

    if eval_params["stitch_threshold"] > 0.0:
        eval_params['do_3D'] = False
    return eval_params


