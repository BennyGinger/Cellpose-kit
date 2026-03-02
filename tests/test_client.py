from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
import numpy as np

import cellpose_kit.workflow.api as api
from cellpose_kit.client import CellposeWrapper


@dataclass
class _BackendStub:
    model_names: list[str]
    init_calls: int = 0

    def init_model(self, user_settings: dict[str, Any], do_denoise: bool) -> Any:
        return _ModelStub()

    def configure_eval_params(self, user_settings: dict[str, Any], use_nuclear_channel: bool, do_denoise: bool) -> dict[str, Any]:
        return {"channels": [1, 2], "do_3D": False}


class _ModelStub:
    def eval(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return ("masks", "flows", "styles", "extra")


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
