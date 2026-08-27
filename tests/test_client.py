from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
import numpy as np

import cellpose_kit.workflow.configuration as configuration
from cellpose_kit.client import CellposeWrapper


@dataclass
class _BackendStub:
    model_names: list[str]
    init_calls: int = 0

    def configure_model(self, user_settings: dict[str, Any], do_denoise: bool) -> dict[str, Any]:
        return {"model_type": user_settings.get("model_type", "cyto2")}

    def supported_settings(self, do_denoise: bool) -> set[str]:
        return {"cellprob_threshold", "model_type"}

    def init_model(self, user_settings: dict[str, Any], do_denoise: bool) -> Any:
        self.init_calls += 1
        return _ModelStub()

    def configure_eval_params(self, user_settings: dict[str, Any], use_nuclear_channel: bool, do_denoise: bool) -> dict[str, Any]:
        return {"channels": [1, 2], "do_3D": False}


class _ModelStub:
    def eval(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return ("masks", "flows", "styles", "extra")


@pytest.fixture(autouse=True)
def _clear_models() -> None:
    CellposeWrapper.clear_models()


def test_cellpose_wrapper_setup_stores_context(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=["cyto2", "cyto3"])

    def _load_backend():
        return backend, "v3"

    monkeypatch.setattr(configuration, "load_backend", _load_backend)

    wrapper = CellposeWrapper(user_settings={"model_type": "cyto2"})
    wrapper.setup()

    assert wrapper._mod_ctx is not None
    assert wrapper.version == "v3"
    assert wrapper.model_names == ["cyto2", "cyto3"]


def test_cellpose_wrapper_run_before_setup_raises() -> None:
    wrapper = CellposeWrapper(user_settings={})
    img = np.zeros((8, 8, 2), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Model context is not set up"):
        wrapper.run(img, "YXC")


def test_cellpose_wrapper_from_dict() -> None:
    settings = {
        'user_settings': {'model_type': 'cyto2'},
        'threading': True,
        'use_nuclear_channel': False,
        'do_denoise': False,
    }
    wrapper = CellposeWrapper.from_dict(settings)
    
    assert wrapper.user_settings == {'model_type': 'cyto2'}
    assert wrapper.threading is True
    assert wrapper.use_nuclear_channel is False
    assert wrapper.do_denoise is False


def test_wrappers_share_only_the_initialized_model(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=["cyto2"])
    monkeypatch.setattr(configuration, "load_backend", lambda: (backend, "v3"))

    first = CellposeWrapper(user_settings={"model_type": "cyto2",
                                           "cellprob_threshold": 0.0,})
    second = CellposeWrapper(user_settings={"model_type": "cyto2",
                                            "cellprob_threshold": 1.0,})
    first.setup()
    second.setup()

    assert first is not second
    assert first._mod_ctx is not second._mod_ctx
    assert first._mod_ctx is not None
    assert second._mod_ctx is not None
    assert first._mod_ctx.model is second._mod_ctx.model
    assert first._mod_ctx.lock is second._mod_ctx.lock
    assert backend.init_calls == 1


def test_unknown_settings_warn_without_failing(monkeypatch: pytest.MonkeyPatch,
                                               caplog: pytest.LogCaptureFixture,
                                               ) -> None:
    backend = _BackendStub(model_names=["cyto2"])
    monkeypatch.setattr(configuration, "load_backend", lambda: (backend, "v3"))

    wrapper = CellposeWrapper(user_settings={"cellprob_treshold": 1.0})
    wrapper.setup()

    assert "Ignoring unsupported Cellpose v3 settings: cellprob_treshold" in caplog.text


def test_clear_models_forces_new_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _BackendStub(model_names=["cyto2"])
    monkeypatch.setattr(configuration, "load_backend", lambda: (backend, "v3"))

    CellposeWrapper(user_settings={}).setup()
    CellposeWrapper.clear_models()
    CellposeWrapper(user_settings={}).setup()

    assert backend.init_calls == 2
