from __future__ import annotations
import logging
from time import perf_counter
from typing import Any

from numpy.typing import NDArray

from cellpose_kit.backend.versioning import SUPPORTED_VERSIONS
from cellpose_kit.workflow.runtime import ModelContext, extract_model_name
from cellpose_kit.workflow.validation import ensure_lists
from cellpose_kit.workflow.array_manip import get_frames_from_array
from cellpose_kit.workflow.models import BatchResult, SegmentationResult
from cellpose_kit.workflow.prep import prepare_batches

logger = logging.getLogger(__name__)


ALLOWED_VERSION = {f"v{ver}" for ver in SUPPORTED_VERSIONS}


def run_cellpose(img: NDArray[Any], axis_order: str, model_context: ModelContext) -> SegmentationResult:
    """
    Run Cellpose segmentation using pre-configured settings.
    
    Policy:
        - v3: Flexible channel input, but must have >= 2 channels if nuclear mode enabled
        - v4: Flexible channel input, but must have 2-3 channels if nuclear mode enabled (2 channels auto-padded to 3)
    
    Parameters:
        img: Input image ndarray
        axis_order: String representing the axis order of the input image (e.g., "TCZYX", "ZYX", "YXC")
        model_context: Configured Cellpose inference context
    Returns:
        SegmentationResult with one BatchResult per independent input batch.
    """
    model = model_context.model
    eval_params = model_context.eval_params
    backend_name = model_context.backend_name

    if backend_name not in ALLOWED_VERSION:
        raise ValueError(f"Invalid backend '{backend_name}'. Expected {ALLOWED_VERSION}. axis_order='{axis_order}', shape={img.shape}, use_nuclear_channel={model_context.use_nuclear_channel}")

    prepared = prepare_batches(img,
                               axis_order,
                               backend=backend_name,
                               use_nuclear_channel=model_context.use_nuclear_channel,
                               do_3D=model_context.do_3D,)

    lock = model_context.lock
    batch_results: list[BatchResult] = []
    start = perf_counter()

    for batch_index, batch in enumerate(prepared.batches):
        frames = get_frames_from_array(batch.array, axis_order=batch.axes)
        frames_count = len(frames)
        if frames_count == 0:
            raise ValueError(
                f"Inference batch {batch_index} has no frames; axes={batch.axes!r}, shape={batch.array.shape}.")

        logger.debug("Running inference batch %s: channel=%s, frames=%s, shape=%s, axes=%s",
                     batch_index,
                     batch.channel_index,
                     frames_count,
                     batch.array.shape,
                     batch.axes,)

        if lock is not None:
            with lock:
                logger.debug("Threading lock acquired, running inference.")
                results = model.eval(frames, **eval_params)
        else:
            logger.debug("No threading lock provided, running inference directly.")
            results = model.eval(frames, **eval_params)

        # masks: list[NDArray], flows: list[list[NDArray]], styles: list[NDArray]]
        masks, flows, styles = ensure_lists(results[:3], frames_count, ("masks", "flows", "styles"))

        batch_results.append(BatchResult(channel_index=batch.channel_index,
                                         masks=masks,
                                         flows=flows,
                                         styles=styles,))

    inference_time_sec = perf_counter() - start
    logger.debug("Cellpose completed: backend=%s, model=%s, batches=%s, output_axes=%s, duration=%.3fs",
                 backend_name,
                 extract_model_name(model_context),
                 len(batch_results),
                 prepared.output_axes,
                 inference_time_sec,)
    return SegmentationResult(batches=batch_results,
                              output_axes=prepared.output_axes,)
