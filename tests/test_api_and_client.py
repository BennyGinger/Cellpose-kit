from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import pytest
import numpy as np

import cellpose_kit.api as api
from cellpose_kit.backend.runtime import ModelContext
from cellpose_kit.client import CellposeWrapper


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


def test_run_cellpose_validates_and_returns_first_three(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ModelStub()
    model_context = ModelContext(model=cast(Any, model), eval_params={"channels": [1, 2]}, backend_name="v3")
    calls: list[tuple[Any, ...]] = []

    def _validate(img, axis_order, eval_params, backend_name):
        calls.append((img, axis_order, eval_params, backend_name))

    monkeypatch.setattr(api, "validate_image_channels", _validate)

    img = np.zeros((8, 8, 2), dtype=np.uint8)
    result = api.run_cellpose(img, "YXC", model_context)

    assert result == ("masks", "flows", "styles")
    assert calls == [(img, "YXC", {"channels": [1, 2]}, "v3")]
    assert model.last_kwargs == {"channels": [1, 2]}


def test_run_cellpose_uses_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ModelStub()
    lock = _LockProbe(Lock())
    model_context = ModelContext(model=cast(Any, model), eval_params={}, backend_name="v3", lock=cast(Any, lock))

    monkeypatch.setattr(api, "validate_image_channels", lambda *args, **kwargs: None)

    img = np.zeros((8, 8, 2), dtype=np.uint8)
    api.run_cellpose(img, "YXC", model_context)

    assert lock.entered is True


def test_cellpose_wrapper_setup_stores_context(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=["cyto2", "cyto3"])

    def _load_backend():
        return backend, "v3"

    monkeypatch.setattr(api, "load_backend", _load_backend)

    wrapper = CellposeWrapper(user_settings={"model_type": "cyto2"})
    wrapper.setup()

    assert wrapper._mod_ctx is not None
    assert wrapper.version == "v3"
    assert wrapper.model_names == ["cyto2", "cyto3"]


def test_cellpose_wrapper_run_before_setup_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = CellposeWrapper(user_settings={})
    img = np.zeros((8, 8, 2), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Model context is not set up"):
        wrapper.run(img, "YXC")
