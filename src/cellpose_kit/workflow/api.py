from __future__ import annotations
import logging
from time import perf_counter
from typing import Any, TYPE_CHECKING
from threading import Lock

from cellpose_kit.workflow.array_manip import get_frames_from_array
from cellpose_kit.workflow.validation import ensure_lists
from numpy.typing import NDArray

from cellpose_kit.workflow.runtime import ModelContext, extract_model_name
from cellpose_kit.backend.factory import load_backend
from cellpose_kit.workflow.models import SegmentationResult, StreamResult
from cellpose_kit.workflow.prep import prepare_streams

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cellpose.denoise import CellposeDenoiseModel

logger = logging.getLogger(__name__)


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

    # Get the 3D flag
    do_3D = bool(eval_params.get("do_3D", False)) or float(eval_params.get("stitch_threshold", 0)) > 0
    
    if model is not None:
        logger.info(f"Cellpose {backend_name} model reused from cache.")
    else:
        logger.info(f"Cellpose {backend_name} model initialized.")

    model_context = ModelContext(model=model_instance, 
                                 eval_params=eval_params,
                                 do_3D=do_3D,
                                 use_nuclear_channel=use_nuclear_channel,
                                 model_names=backend.model_names, 
                                 backend_name=backend_name)

    if threading:
        logger.info("Threading enabled: Adding lock for thread-safe model inference")
        model_context.lock = Lock()

    return model_context


def run_cellpose(img: NDArray[Any], axis_order: str, model_context: ModelContext) -> SegmentationResult:
    """
    Run Cellpose segmentation using pre-configured settings.
    
    Policy:
        - v3: Flexible channel input, but must have >= 2 channels if nuclear mode enabled
        - v4: Flexible channel input, but must have 2-3 channels if nuclear mode enabled (2 channels auto-padded to 3)
    
    Parameters:
        img: Input image ndarray
        axis_order: String representing the axis order of the input image (e.g., "TCZYX", "ZYX", "YXC")
        model_context: Output from setup_cellpose()
        split_channels: If True and use_nuclear_channel=False, split C axis into channel streams

    Returns:
        SegmentationResult with one StreamResult per prepared input stream.
    """
    model = model_context.model
    eval_params = model_context.eval_params
    backend_name = model_context.backend_name

    if backend_name not in {"v3", "v4"}:
        raise ValueError(f"Invalid backend '{backend_name}'. Expected 'v3' or 'v4'. axis_order='{axis_order}', shape={img.shape}, use_nuclear_channel={model_context.use_nuclear_channel}")

    streams, run_meta = prepare_streams(img,
                                        axis_order,
                                        backend=backend_name,
                                        use_nuclear_channel=model_context.use_nuclear_channel,
                                        do_3D=model_context.do_3D)

    lock = model_context.lock
    stream_results: list[StreamResult] = []
    frames_per_stream: list[int] = []
    start = perf_counter()

    for stream in streams:
        frames = get_frames_from_array(stream.source_array, axis_order=stream.axis_order)
        frames_count = len(frames)
        if frames_count == 0:
            raise ValueError(f"Prepared stream '{stream.stream_id}' has no frames. axis_order='{stream.axis_order}', shape={stream.source_array.shape}")

        if lock is not None:
            with lock:
                logger.debug("Threading lock acquired, running inference.")
                results = model.eval(frames, **eval_params)
        else:
            logger.debug("No threading lock provided, running inference directly.")
            results = model.eval(frames, **eval_params)

        # masks: list[NDArray], flows: list[list[NDArray]], styles: list[NDArray]]
        masks, flows, styles = ensure_lists(results[:3], frames_count, ("masks", "flows", "styles"))

        stream_meta = dict(stream.meta)
        stream_meta["frames_count"] = frames_count

        stream_results.append(StreamResult(stream_id=stream.stream_id,
                                           channel_index=stream.meta.get("channel_index"),
                                           masks=masks,
                                           flows=flows,
                                           styles=styles,
                                           meta=stream_meta,))
        frames_per_stream.append(frames_count)

    inference_time_sec = perf_counter() - start
    any_padding_applied = any(bool(s.meta.get("padded_to_3", False)) for s in streams)

    result_meta: dict[str, Any] = {**run_meta,
                                    "model_name": extract_model_name(model_context),
                                    "n_streams": len(stream_results),
                                    "frames_per_stream": frames_per_stream,
                                    "any_padding_applied": any_padding_applied,
                                    "inference_time_sec": inference_time_sec,}

    return SegmentationResult(streams=stream_results, meta=result_meta)



