from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    def single(self) -> StreamResult:
        """Return the single stream result. Raises ValueError if there are multiple streams."""
        if len(self.streams) != 1:
            raise ValueError(f"Expected exactly 1 stream, got {len(self.streams)}")
        return self.streams[0]

    def masks_by_channel(self) -> dict[int, list[NDArray[Any]]]:
        result: dict[int, list[NDArray[Any]]] = {}
        for stream in self.streams:
            channel_idx = stream.channel_index if stream.channel_index is not None else 0
            result[channel_idx] = stream.masks
        return result