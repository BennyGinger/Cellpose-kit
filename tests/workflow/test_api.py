from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import pytest
import numpy as np

import cellpose_kit.workflow.api as api
from cellpose_kit.workflow.runtime import ModelContext
from cellpose_kit.workflow.models import InputStream


@dataclass
class _BackendStub:
    model_names: list[str]
    init_calls: int = 0
    eval_calls: int = 0

    def init_model(self, user_settings: dict[str, Any], do_denoise: bool) -> Any:
        self.init_calls += 1
        return _ModelStub()

    def configure_eval_params(self, user_settings: dict[str, Any], use_nuclear_channel: bool, do_denoise: bool) -> dict[str, Any]:
        return {"channels": [1, 2], "do_3D": False}


class _ModelStub:
    def __init__(self) -> None:
        self.last_args: tuple[Any, ...] | None = None
        self.last_kwargs: dict[str, Any] | None = None

    def eval(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        self.last_args = args
        self.last_kwargs = kwargs
        return ("masks", "flows", "styles", "extra")


class _LockProbe:
    def __init__(self, lock: Lock) -> None:
        self.lock = lock
        self.entered = False

    def __enter__(self) -> None:
        self.entered = True
        self.lock.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.lock.release()
        return None


def test_setup_cellpose_uses_existing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=["cyto2"])

    def _load_backend():
        return backend, "v3"

    monkeypatch.setattr(api, "load_backend", _load_backend)
    existing_model = _ModelStub()

    model_context = api.setup_cellpose({"model_type": "cyto2"}, model=cast(Any, existing_model))

    assert model_context.model is existing_model
    assert backend.init_calls == 0
    assert model_context.backend_name == "v3"


def test_setup_cellpose_creates_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=[])

    def _load_backend():
        return backend, "v4"

    monkeypatch.setattr(api, "load_backend", _load_backend)

    model_context = api.setup_cellpose({}, threading=True)

    assert model_context.lock is not None


def test_run_cellpose_returns_stream_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ModelStub()
    model_context = ModelContext(model=cast(Any, model), eval_params={"channels": [1, 2]}, backend_name="v3")
    model_context.use_nuclear_channel = False

    def _prepare_streams(img, axis_order, backend, use_nuclear_channel):
        stream = InputStream(
            source_array=img,
            axis_order=axis_order,
            stream_id="stream0",
            meta={
                "channel_index": None,
                "padded_to_3": False,
                "original_n_channels": 2,
                "stream_shape": tuple(img.shape),
                "stream_axis_order": axis_order,
            },
        )
        return [stream], {
            "backend": backend,
            "use_nuclear_channel": use_nuclear_channel,
            "split_channels": False,
            "input_axis_order": axis_order,
            "input_shape": tuple(img.shape),
            "n_input_channels": 2,
            "any_padding_applied": False,
        }

    monkeypatch.setattr(api, "prepare_streams", _prepare_streams)

    img = np.zeros((8, 8, 2), dtype=np.uint8)
    result = api.run_cellpose(img, "YXC", model_context)

    assert len(result.streams) == 1
    assert result.single().masks == ["masks"]
    assert result.single().flows == ["flows"]
    assert result.single().styles == ["styles"]
    assert result.meta["n_streams"] == 1
    assert model.last_kwargs == {"channels": [1, 2]}
    assert isinstance(model.last_args, tuple)
    assert len(model.last_args) == 1
    assert isinstance(model.last_args[0], list)
    assert len(model.last_args[0]) == 1


def test_run_cellpose_uses_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ModelStub()
    lock = _LockProbe(Lock())
    model_context = ModelContext(model=cast(Any, model), eval_params={}, backend_name="v3", lock=cast(Any, lock))
    model_context.use_nuclear_channel = False

    def _prepare_streams(img, axis_order, backend, use_nuclear_channel):
        return [
            InputStream(
                source_array=img,
                axis_order=axis_order,
                stream_id="stream0",
                meta={
                    "channel_index": None,
                    "padded_to_3": False,
                    "original_n_channels": 2,
                    "stream_shape": tuple(img.shape),
                    "stream_axis_order": axis_order,
                },
            )
        ], {
            "backend": backend,
            "use_nuclear_channel": use_nuclear_channel,
            "split_channels": False,
            "input_axis_order": axis_order,
            "input_shape": tuple(img.shape),
            "n_input_channels": 2,
            "any_padding_applied": False,
        }

    monkeypatch.setattr(api, "prepare_streams", _prepare_streams)

    img = np.zeros((8, 8, 2), dtype=np.uint8)
    api.run_cellpose(img, "YXC", model_context)

    assert lock.entered is True
