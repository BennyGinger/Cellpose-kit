from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cellpose_kit.workflow.array_manip import pad_to_3_channels, split_channels
from cellpose_kit.workflow.models import InferenceBatch, PreparedInput
from cellpose_kit.workflow.utils import count_axis, get_axis
from cellpose_kit.workflow.validation import validate_array


logger = logging.getLogger(__name__)


def prepare_batches(img: NDArray[Any],
                    axis_order: str,
                    backend: str,
                    use_nuclear_channel: bool,
                    do_3D: bool,
                    ) -> PreparedInput:
    """
    Validate and prepare independent Cellpose inference batches.
    """
    validate_array(img, axis_order, backend, use_nuclear_channel, do_3D)
    channel_axis = get_axis(axis_order, "C")
    channel_count = count_axis(img, axis_order, "C")
    output_axes = axis_order

    logger.debug("Preparing Cellpose input: shape=%s, axes=%s, backend=%s",
                 img.shape,
                 axis_order,
                 backend,)

    if use_nuclear_channel:
        prepared = img
        output_axes = output_axes.replace("C", "")
        if backend == "v4" and channel_count == 2:
            logger.debug("Padding v4 nuclear input from two to three channels.")
            prepared = pad_to_3_channels(img, axis_order)
        batches = [InferenceBatch(array=prepared, axes=axis_order)]
    elif channel_axis is not None and channel_count > 1:
        arrays, batch_axes = split_channels(img, axis_order)
        batches = [InferenceBatch(array=array,
                                  axes=batch_axes,
                                  channel_index=index,)
                   for index, array in enumerate(arrays)]
        logger.debug("Processing %s channels as independent inference batches.",
                     channel_count,)
    else:
        if channel_axis is None:
            prepared = img
            batch_axes = axis_order
        else:
            prepared = np.take(img, 0, axis=channel_axis)
            batch_axes = axis_order.replace("C", "", 1)
            output_axes = batch_axes
            logger.debug("Removed singleton C axis for Cellpose inference.")
        batches = [InferenceBatch(array=prepared, axes=batch_axes)]

    logger.debug("Prepared %s inference batch(es); output axes=%s.",
                 len(batches),
                 output_axes,)
    return PreparedInput(batches=batches, output_axes=output_axes)
