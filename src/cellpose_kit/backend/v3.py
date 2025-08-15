from cellpose.models import MODEL_NAMES, normalize_default


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