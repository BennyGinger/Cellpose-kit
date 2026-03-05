from __future__ import annotations

from typing import Any, TypeVar
import logging

from cellpose_kit.workflow.array_manip import pad_to_3_channels, split_channels
from cellpose_kit.workflow.utils import count_axis, get_axis
from cellpose_kit.workflow.validation import validate_array
import numpy as np
from numpy.typing import NDArray

from cellpose_kit.workflow.models import InputStream


T = TypeVar('T', bound=np.generic)

logger = logging.getLogger(__name__)

def prepare_streams(img: NDArray[Any], axis_order: str, backend: str, use_nuclear_channel: bool, do_3D: bool) -> tuple[list[InputStream], dict[str, Any]]:
    """
    Prepare input streams for Cellpose inference.
    
    Validates channel requirements, applies necessary transformations (padding/splitting),
    and creates InputStream objects with metadata.
    
    Parameters:
        img: Input image array
        axis_order: String representing the axis order (e.g., "TCZYX", "YXC")
        backend: Cellpose backend ("v3" or "v4")
        use_nuclear_channel: Whether nuclear channel mode is enabled
        do_3D: Whether 3D mode is enabled
        
    Returns:
        Tuple of (list of InputStreams, run metadata dict)
    """
    # Validate requirements upfront
    validate_array(img, axis_order, backend, use_nuclear_channel, do_3D)
    
    channel_axis = get_axis(axis_order, "C")
    n_channels = count_axis(img, axis_order, "C")

    streams: list[InputStream] = []
    is_padding_applied = False
    channel_split = False
    out_axis_order = axis_order

    if use_nuclear_channel:
        # Nuclear mode: keep channels together
        prepared = img # At this point img has 2 or 3 channels as validated above
        padded_to_3 = False
        # Remove C from axis order as output will be single mask channel regardless of input channels in nuclear mode
        out_axis_order = out_axis_order.replace("C", "")

        if backend == "v4" and n_channels == 2:
            prepared = pad_to_3_channels(img, axis_order)
            padded_to_3 = True
            is_padding_applied = True

        stream_meta = {"channel_index": None,
                       "padded_to_3": padded_to_3,
                       "original_n_channels": n_channels,
                       "stream_shape": prepared.shape,
                       "stream_axis_order": axis_order,}
        
        streams.append(InputStream(source_array=prepared,
                                   axis_order=axis_order,
                                   stream_id="stream0",
                                   meta=stream_meta,))
        
    else: # Non-nuclear mode
        # Split channels
        if channel_axis is not None and n_channels > 1:
            split_arrays, split_axis_order = split_channels(img, axis_order)
            channel_split = True
            
            for idx, array in enumerate(split_arrays):
                stream_meta = {"channel_index": idx,
                               "padded_to_3": False,
                               "original_n_channels": n_channels,
                               "stream_shape": array.shape,
                               "stream_axis_order": split_axis_order,}
                
                streams.append(InputStream(source_array=array,
                                           axis_order=split_axis_order,
                                           stream_id=f"stream{idx}",
                                           meta=stream_meta,))
                
        else: # No splitting: single stream with original array
            # Remove C axis as only one channel will be segmented
            out_axis_order = out_axis_order.replace("C", "") 
            
            stream_meta = {"channel_index": None,
                           "padded_to_3": False,
                           "original_n_channels": n_channels,
                           "stream_shape": img.shape,
                           "stream_axis_order": out_axis_order,}
            
            streams.append(InputStream(source_array=img,
                                       axis_order=out_axis_order,
                                       stream_id="stream0",
                                       meta=stream_meta,))

    run_meta_partial = {"backend": backend,
                        "use_nuclear_channel": use_nuclear_channel,
                        "split_channels": channel_split,
                        "input_axis_order": axis_order,
                        "output_axis_order": out_axis_order,
                        "input_shape": img.shape,
                        "n_input_channels": n_channels,
                        "any_padding_applied": is_padding_applied,
                        "do_3D": do_3D,}
    logger.debug(f"Prepared {len(streams)} stream(s) with meta: {run_meta_partial}")
    return streams, run_meta_partial
