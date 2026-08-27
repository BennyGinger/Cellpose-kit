from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class InferenceBatch:
    array: NDArray[Any]
    axes: str
    channel_index: int | None = None


@dataclass
class PreparedInput:
    batches: list[InferenceBatch]
    output_axes: str


@dataclass
class BatchResult:
    channel_index: int | None
    masks: list[NDArray[Any]]
    flows: list[list[NDArray[Any]]]
    styles: list[NDArray[Any]]


@dataclass
class SegmentationResult:
    batches: list[BatchResult]
    output_axes: str

    @property
    def masks(self) -> list[list[NDArray[Any]]]:
        """
        Return masks in inference-batch order.
        """
        return [batch.masks for batch in self.batches]

    @property
    def flows(self) -> list[list[list[NDArray[Any]]]]:
        """
        Return flows in inference-batch order.
        """
        return [batch.flows for batch in self.batches]

    @property
    def styles(self) -> list[list[NDArray[Any]]]:
        """
        Return styles in inference-batch order.
        """
        return [batch.styles for batch in self.batches]

    def masks_array(self) -> NDArray[Any]:
        """
        Reconstruct the mask array using the output axes.
        """
        has_time = "T" in self.output_axes
        has_channels = "C" in self.output_axes

        def stack_time(masks: list[NDArray[Any]]) -> NDArray[Any]:
            if has_time:
                return np.stack(masks, axis=self.output_axes.index("T"))
            if len(masks) != 1:
                raise ValueError(
                    f"Expected one mask without a T axis, got {len(masks)}.")
            return masks[0]

        batch_arrays = [(batch.channel_index, stack_time(batch.masks))
                        for batch in self.batches]
        if not has_channels:
            if len(batch_arrays) != 1:
                raise ValueError(
                    f"Expected one inference batch without a C axis, got {len(batch_arrays)}.")
            return batch_arrays[0][1]

        if any(channel is None for channel, _ in batch_arrays):
            raise ValueError("Cannot reconstruct C because a batch has no channel index.")
        batch_arrays.sort(key=lambda item: int(item[0]))  # type: ignore[arg-type]
        return np.stack([array for _, array in batch_arrays],
                        axis=self.output_axes.index("C"),)
