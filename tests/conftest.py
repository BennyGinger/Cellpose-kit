from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest


class _DummyModel:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def eval(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return ()


def _install_fake_cellpose() -> None:
    cellpose = ModuleType("cellpose")
    setattr(cellpose, "__version__", "4.0.0")

    models = ModuleType("cellpose.models")
    setattr(models, "CellposeModel", _DummyModel)
    setattr(models, "normalize_default", {"dummy": True})
    setattr(models, "MODEL_NAMES", ["cyto2", "cyto3", "cpsam"])

    denoise = ModuleType("cellpose.denoise")
    setattr(denoise, "CellposeDenoiseModel", _DummyModel)

    io = ModuleType("cellpose.io")

    def logger_setup() -> None:
        return None

    setattr(io, "logger_setup", logger_setup)

    sys.modules["cellpose"] = cellpose
    sys.modules["cellpose.models"] = models
    sys.modules["cellpose.denoise"] = denoise
    sys.modules["cellpose.io"] = io


@pytest.fixture
def backend_v3() -> ModuleType:
    _install_fake_cellpose()
    module = importlib.import_module("cellpose_kit.backend.v3")
    importlib.reload(module)
    return module


@pytest.fixture
def backend_v4() -> ModuleType:
    _install_fake_cellpose()
    module = importlib.import_module("cellpose_kit.backend.v4")
    importlib.reload(module)
    return module
