from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import pytest
import numpy as np

import cellpose_kit.workflow.api as api
import cellpose_kit.workflow.configuration as configuration
from cellpose_kit.workflow.runtime import ModelContext
from cellpose_kit.workflow.models import InferenceBatch, PreparedInput


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


def test_initialize_model_uses_existing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=["cyto2"])

    def _load_backend():
        return backend, "v3"

    monkeypatch.setattr(configuration, "load_backend", _load_backend)
    existing_model = _ModelStub()

    model_context = configuration.initialize_model({"model_type": "cyto2"},
                                                   model=cast(Any, existing_model),)

    assert model_context.model is existing_model
    assert backend.init_calls == 0
    assert model_context.backend_name == "v3"


def test_initialize_model_creates_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=[])

    def _load_backend():
        return backend, "v4"

    monkeypatch.setattr(configuration, "load_backend", _load_backend)

    model_context = configuration.initialize_model({}, threading=True)

    assert model_context.lock is not None


def test_configure_inference_reuses_model_and_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=["cyto2"])

    def _load_backend():
        return backend, "v3"

    monkeypatch.setattr(configuration, "load_backend", _load_backend)
    existing_model = _ModelStub()
    existing_lock = Lock()
    model_context = ModelContext(model=cast(Any, existing_model),
                                 eval_params={},
                                 backend_name="v3",
                                 lock=existing_lock,)

    configured = configuration.configure_inference({"cellprob_threshold": 1.0},
                                                    model_context,)

    assert configured.model is existing_model
    assert configured.lock is existing_lock
    assert backend.init_calls == 0


def test_run_cellpose_returns_stream_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ModelStub()
    model_context = ModelContext(model=cast(Any, model), eval_params={"channels": [1, 2]}, backend_name="v3")
    model_context.use_nuclear_channel = False

    def _prepare_batches(img, axis_order, backend, use_nuclear_channel, do_3D):
        return PreparedInput(
            batches=[InferenceBatch(array=img, axes=axis_order)],
            output_axes=axis_order,)

    monkeypatch.setattr(api, "prepare_batches", _prepare_batches)

    img = np.zeros((8, 8, 2), dtype=np.uint8)
    result = api.run_cellpose(img, "YXC", model_context)

    assert len(result.batches) == 1
    assert result.batches[0].masks == ["masks"]
    assert result.batches[0].flows == ["flows"]
    assert result.batches[0].styles == ["styles"]
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

    def _prepare_batches(img, axis_order, backend, use_nuclear_channel, do_3D):
        return PreparedInput(
            batches=[InferenceBatch(array=img, axes=axis_order)],
            output_axes=axis_order,)

    monkeypatch.setattr(api, "prepare_batches", _prepare_batches)

    img = np.zeros((8, 8, 2), dtype=np.uint8)
    api.run_cellpose(img, "YXC", model_context)

    assert lock.entered is True
