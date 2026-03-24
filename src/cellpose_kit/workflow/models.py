from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class InputStream:
    source_array: NDArray[Any]
    axis_order: str
    stream_id: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamResult:
    stream_id: str
    channel_index: int | None
    masks: list[NDArray[Any]]
    flows: list[list[NDArray[Any]]]
    styles: list[NDArray[Any]]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    streams: list[StreamResult]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def masks(self) -> list[list[NDArray[Any]]]:
        """Return per-stream masks in stream order."""
        return [stream.masks for stream in self.streams]

    @property
    def flows(self) -> list[list[list[NDArray[Any]]]]:
        """Return per-stream flows in stream order."""
        return [stream.flows for stream in self.streams]

    @property
    def styles(self) -> list[list[NDArray[Any]]]:
        """Return per-stream styles in stream order."""
        return [stream.styles for stream in self.streams]

    @property
    def output_axis_order(self) -> str | None:
        return self.meta.get("output_axis_order")
    
    def masks_array(self) -> NDArray[Any]:
        """
        Reconstruct a mask array according to `output_axis_order`.

        Contract:
        - Each stream.masks is a list of per-T-frame masks (T order).
        - If 'T' is in output_axis_order, stack frames along that axis.
          Otherwise expect exactly 1 frame per stream.
        - If 'C' is in output_axis_order, stack streams (sorted by channel_index) along that axis.
          Otherwise expect exactly 1 stream.
        """
        out = self.output_axis_order
        if out is None:
            raise ValueError("SegmentationResult.meta has no 'output_axis_order'")

        has_t = "T" in out
        has_c = "C" in out

        # Helper: stack one stream across time (if applicable)
        def _stack_time(masks: list[NDArray[Any]]) -> NDArray[Any]:
            if has_t:
                t_axis = out.index("T")
                # masks already in T order
                return np.stack(masks, axis=t_axis)
            # no T in output -> must be single frame
            if len(masks) != 1:
                raise ValueError(f"Expected 1 mask (no 'T' in output_axis_order='{out}'), got {len(masks)}")
            return masks[0]

        # 1) build per-stream arrays (time-stacked if needed)
        per_stream = [(s.channel_index, _stack_time(s.masks)) for s in self.streams]

        # 2) combine channels if needed
        if not has_c:
            if len(per_stream) != 1:
                raise ValueError(f"Expected 1 stream (no 'C' in output_axis_order='{out}'), got {len(per_stream)}")
            return per_stream[0][1]

        # has C -> sort by channel_index and stack on C axis
        if any(ch is None for ch, _arr in per_stream):
            raise ValueError("Cannot restack along 'C' because at least one stream has channel_index=None")

        per_stream_sorted = sorted(per_stream, key=lambda x: int(x[0]))  # type: ignore[arg-type]
        channel_arrays = [arr for _ch, arr in per_stream_sorted]

        c_axis = out.index("C")
        # This inserts a new axis at the correct position
        return np.stack(channel_arrays, axis=c_axis)